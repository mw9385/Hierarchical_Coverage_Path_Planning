import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, n_head, qkv_fc_layer, fc_layer):
        # qkv_fc_layer = [d_embed, d_model]
        # fc_layer = [d_model, d_embed]
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.query_fc_layer = copy.deepcopy(qkv_fc_layer)
        self.key_fc_layer = copy.deepcopy(qkv_fc_layer)
        self.value_fc_layer = copy.deepcopy(qkv_fc_layer)
        self.fc_layer = fc_layer

    def forward(self, query, key, value, mask = None):
        # query, key, value = [batch, seq_len, n_hidden]
        # mask = [batch, seq_len, seq_len]
        n_batch = query.shape[0] 

        # reshape [batch, seq_len, n_hidden] to [batch, n_head, seq_len, d_model]
        def transform(x, fc_layer):
            # x = [batch, seq_len, n_hidden]
            out = fc_layer(x) # n_hidden -> d_model, out = [batch, seq_len, d_model]
            out = out.view(n_batch, -1, self.n_head, self.d_model//self.n_head) 
            # out size = [batch, seq_len, n_head, d_k], d_k = d_model/n_head
            out = out.transpose(1,2) # out = [batch, n_head, seq_len, d_k]
            return out
        
        query = transform(query, self.query_fc_layer)
        key = transform(key, self.key_fc_layer)
        value = transform(value, self.value_fc_layer)

        if mask is not None:
            mask = mask.unsqueeze(1)
        out = self.calcuclate_attention(query, key, value, mask) #out [batch, n_head, seq_len, d_k]
        out = out.transpose(1,2) # out = [batch, seq_len, n_head, d_k]
        out = out.contiguous().view(n_batch, -1, self.d_model) # out = [batch, seq_len, d_model]
        out = self.fc_layer(out) # d_model -> d_embed. out = [batch, seq_len, d_embed]
        return out

    def calcuclate_attention(self, query, key, value, mask):
        d_k = key.size(-1) # query, key, value = [batch, n_head, seq_len, d_k]
        score = torch.matmul(query, key.transpose(-2,-1)) # Q x K^T
        score = score/ math.sqrt(d_k) # scailing        
        if mask is not None:
            # tensor.masked_fill(mask_index, value): Fills the elements with value where mask is True
            score = score.masked_fill(mask == 0, -1e9) 
        out = F.softmax(score, dim = -1) # get the softmax score
        out = torch.matmul(out, value) # score x V
        return out

class Attention(nn.Module):
    def __init__(self, n_hidden):
        super(Attention, self).__init__()
        self.size = 0
        self.batch_size = 0
        self.dim = n_hidden
        
        v  = torch.FloatTensor(n_hidden).cuda()
        self.v  = nn.Parameter(v)
        self.v.data.uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden))
        
        # parameters for pointer attention
        self.Wref = nn.Linear(n_hidden, n_hidden)
        self.Wq = nn.Linear(n_hidden, n_hidden)
    
    
    def forward(self, q, ref):       # query and reference
        self.batch_size = q.size(0)
        self.size = int(ref.size(0) / self.batch_size)
        q = self.Wq(q)     # (B, dim)
        ref = self.Wref(ref)
        ref = ref.view(self.batch_size, self.size, self.dim)  # (B, size, dim)
        
        q_ex = q.unsqueeze(1).repeat(1, self.size, 1) # (B, size, dim)
        # v_view: (B, dim, 1)
        v_view = self.v.unsqueeze(0).expand(self.batch_size, self.dim).unsqueeze(2)
        
        # (B, size, dim) * (B, dim, 1)
        u = torch.bmm(torch.tanh(q_ex + ref), v_view).squeeze(2)
        
        return u, ref

class LSTM(nn.Module):
    def __init__(self, n_hidden):
        super(LSTM, self).__init__()
        
        # parameters for input gate
        self.Wxi = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Whi = nn.Linear(n_hidden, n_hidden)    # W(ht)
        self.wci = nn.Linear(n_hidden, n_hidden)    # w(ct)
        
        # parameters for forget gate
        self.Wxf = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Whf = nn.Linear(n_hidden, n_hidden)    # W(ht)
        self.wcf = nn.Linear(n_hidden, n_hidden)    # w(ct)
        
        # parameters for cell gate
        self.Wxc = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Whc = nn.Linear(n_hidden, n_hidden)    # W(ht)
        
        # parameters for forget gate
        self.Wxo = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Who = nn.Linear(n_hidden, n_hidden)    # W(ht)
        self.wco = nn.Linear(n_hidden, n_hidden)    # w(ct)
    
    
    def forward(self, x, h, c):       # query and reference
        
        # input gate
        i = torch.sigmoid(self.Wxi(x) + self.Whi(h) + self.wci(c))
        # forget gate
        f = torch.sigmoid(self.Wxf(x) + self.Whf(h) + self.wcf(c))
        # cell gate
        c = f * c + i * torch.tanh(self.Wxc(x) + self.Whc(h))
        # output gate
        o = torch.sigmoid(self.Wxo(x) + self.Who(h) + self.wco(c))
        
        h = o * torch.tanh(c)
        
        return h, c

class Attention(nn.Module):
    def __init__(self, n_hidden):
        super(Attention, self).__init__()
        self.size = 0
        self.batch_size = 0
        self.dim = n_hidden
        
        v  = torch.FloatTensor(n_hidden).cuda()
        self.v  = nn.Parameter(v)
        self.v.data.uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden))
        
        # parameters for pointer attention
        self.Wref = nn.Linear(n_hidden, n_hidden)
        self.Wq = nn.Linear(n_hidden, n_hidden)
    
    
    def forward(self, q, ref):       # query and reference
        self.batch_size = q.size(0)
        self.size = int(ref.size(0) / self.batch_size)
        q = self.Wq(q)     # (B, dim)
        ref = self.Wref(ref)
        ref = ref.view(self.batch_size, self.size, self.dim)  # (B, size, dim)
        
        q_ex = q.unsqueeze(1).repeat(1, self.size, 1) # (B, size, dim)
        # v_view: (B, dim, 1)
        v_view = self.v.unsqueeze(0).expand(self.batch_size, self.dim).unsqueeze(2)
        
        # (B, size, dim) * (B, dim, 1)
        u = torch.bmm(torch.tanh(q_ex + ref), v_view).squeeze(2)
        
        return u, ref


class Encoder(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, high_level):
        super(Encoder, self).__init__()
        cp = lambda x: copy.deepcopy(x)
        self.embedding_x = nn.Linear(n_feature, n_hidden).cuda()        
        self.high_level = high_level
        self.multi_head_attention = MultiHeadAttentionLayer(
                                            d_model= n_hidden,
                                            n_head = 8,
                                            qkv_fc_layer= nn.Linear(n_hidden, n_hidden),
                                            fc_layer = nn.Linear(n_hidden, n_hidden)).cuda()        
        self.low_attention = cp(self.multi_head_attention)
        self.high_attention = cp(self.multi_head_attention)        

    def forward(self, batch_data, high_mask = None, low_mask = None):
        self.batch_data = batch_data        
        self.low_node = []
        self.embedded_x = torch.tensor(()).cuda()        
        for batch_samples in self.batch_data:
            _x = torch.tensor(()).cuda()
            _low_node = []
            for sample in batch_samples:
                x = self.embedding_x(sample.cuda()).unsqueeze(0)
                _low_node.append(self.low_attention(query = x, key = x, value = x, mask = None))                
            self.low_node.append(_low_node)                                
        
        # cell embedding
        self.high_node = self.cell_embedding(mask = None)

        return [self.low_node, self.high_node]

    def cell_embedding(self, mask = None):
        # calculate the mean of the embedded cells. This process will calculate the correlation between cells
        # embedded_size = [batch, n_cells, n_hidden]
        embedded_mean = torch.tensor(()).cuda()
        for sub_nodes in self.low_node:
            _mean = torch.tensor(()).cuda()
            for cell_samples in sub_nodes:
                _mean = torch.cat((_mean, torch.mean(cell_samples, dim=1)), dim = 0) 
            embedded_mean = torch.cat((embedded_mean, _mean.unsqueeze(0)), dim = 0)
        att_cell = self.high_attention(query = embedded_mean, key = embedded_mean, value = embedded_mean, mask = None)
        return att_cell


class GPN(torch.nn.Module):
    
    def __init__(self, n_feature, n_hidden):
        super(GPN, self).__init__()
        self.city_size = 0
        self.batch_size = 0
        self.dim = n_hidden
        
        # lstm for first turn
        self.lstm0 = nn.LSTM(n_hidden, n_hidden)
        
        # pointer layer
        self.pointer = Attention(n_hidden)
        
        # lstm encoder
        self.encoder = LSTM(n_hidden)
        
        # trainable first hidden input
        h0 = torch.FloatTensor(n_hidden).cuda()
        c0 = torch.FloatTensor(n_hidden).cuda()
        
        # trainable latent variable coefficient
        alpha = torch.ones(1).cuda()
        
        self.h0 = nn.Parameter(h0)
        self.c0 = nn.Parameter(c0)
        
        self.alpha = nn.Parameter(alpha)
        self.h0.data.uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden))
        self.c0.data.uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden))
        
        r1 = torch.ones(1).cuda()
        r2 = torch.ones(1).cuda()
        r3 = torch.ones(1).cuda()
        self.r1 = nn.Parameter(r1)
        self.r2 = nn.Parameter(r2)
        self.r3 = nn.Parameter(r3)
                        
        # weights for GNN
        self.W1 = nn.Linear(n_hidden, n_hidden)
        self.W2 = nn.Linear(n_hidden, n_hidden)
        self.W3 = nn.Linear(n_hidden, n_hidden)
        
        # aggregation function for GNN
        self.agg_1 = nn.Linear(n_hidden, n_hidden)
        self.agg_2 = nn.Linear(n_hidden, n_hidden)
        self.agg_3 = nn.Linear(n_hidden, n_hidden)
    
    
    def forward(self, x, X_all, mask, h=None, c=None, latent=None):
        '''
        Inputs (B: batch size, size: city size, dim: hidden dimension)
        
        x: current city coordinate (B, 2)
        X_all: all cities' cooridnates (B, size, 2)
        mask: mask visited cities
        h: hidden variable (B, dim)
        c: cell gate (B, dim)
        latent: latent pointer vector from previous layer (B, size, dim)
        
        Outputs
        
        softmax: probability distribution of next city (B, size)
        h: hidden variable (B, dim)
        c: cell gate (B, dim)
        latent_u: latent pointer vector for next layer
        '''
        
        self.batch_size = X_all.size(0)
        self.city_size = X_all.size(1)
        context = X_all
        
        # =============================
        # process hidden variable
        # =============================
        
        first_turn = False
        if h is None or c is None:
            first_turn = True
        
        if first_turn:
            # (dim) -> (B, dim)
            
            h0 = self.h0.unsqueeze(0).expand(self.batch_size, self.dim)
            c0 = self.c0.unsqueeze(0).expand(self.batch_size, self.dim)

            h0 = h0.unsqueeze(0).contiguous()
            c0 = c0.unsqueeze(0).contiguous()
            
            input_context = context.permute(1,0,2).contiguous()
            _, (h_enc, c_enc) = self.lstm0(input_context, (h0, c0))
            
            # let h0, c0 be the hidden variable of first turn
            h = h_enc.squeeze(0)
            c = c_enc.squeeze(0)
        
        
        # =============================
        # graph neural network encoder
        # =============================
        
        # (B, size, dim)
        context = context.view(-1, self.dim)
        
        context = self.r1 * self.W1(context)\
            + (1-self.r1) * F.relu(self.agg_1(context/(self.city_size-1)))

        context = self.r2 * self.W2(context)\
            + (1-self.r2) * F.relu(self.agg_2(context/(self.city_size-1)))
        
        context = self.r3 * self.W3(context)\
            + (1-self.r3) * F.relu(self.agg_3(context/(self.city_size-1)))
        
        
        # LSTM encoder
        h, c = self.encoder(x, h, c)
        
        # query vector
        q = h
        
        # pointer
        u, _ = self.pointer(q, context)
        
        latent_u = u.clone()
        u = 10 * torch.tanh(u) + mask
        
        if latent is not None:
            u += self.alpha * latent
    
        return F.softmax(u, dim=1), h, c, latent_u
