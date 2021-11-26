import copy
import math
import torch
from torch._C import dtype
from math import sqrt
import torch.nn as nn
import torch.nn.functional as F


class Pointer(nn.Module):
    def __init__(self,
                n_embedding,
                n_hidden,                
                C):
        super(Pointer, self).__init__()
        self.n_embedding = n_embedding
        self.n_hidden = n_hidden        
        self.C = C

        self.W_q = nn.Linear(self.n_embedding, self.n_hidden)
        self.W_k = nn.Linear(self.n_embedding, self.n_hidden)
        # self.W_v = nn.Linear(self.n_embedding, self.n_hidden)

    def forward(self, query, target, mask = None):
        """
        query = [batch, embedding_size]
        target = [batch, seq_len, embedding_size]
        mask = [batch, seq_len]
        """
        q = self.W_q(query) # query size = [batch, n_hidden]
        k = self.W_k(target) # key size = [batch, seq_len, n_hidden]
        # v = self.W_v(target) # value size = [batch, seq_len, n_hidden]
        qk = torch.einsum("ik, ijk -> ij", [q,k]) # qk size = [batch, seq_len]
        qk = self.C * torch.tanh(qk) # qk size = [batch, seq_len]                
        if mask is not None:                    
            qk = torch.masked_fill(qk, mask==1, -1e9)
            # qk.masked_fill(mask==1, -10000) # 이렇게 하면 안되는데 이유가 뭐지?                 
        alpha = torch.softmax(qk, dim = -1)
        return alpha

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, n_hidden, n_head):
        # qkv_fc_layer = [d_embed, d_model]
        # fc_layer = [d_model, d_embed]
        super(MultiHeadAttentionLayer, self).__init__()
        self.n_hidden = n_hidden
        self.n_head = n_head
        self.query_fc_layer = nn.Linear(self.n_hidden, self.n_hidden)
        self.key_fc_layer = nn.Linear(self.n_hidden, self.n_hidden)
        self.value_fc_layer = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc_layer = nn.Linear(self.n_hidden, self.n_hidden)

    def forward(self, query, key, value, mask = None):
        # query, key, value = [batch, seq_len, n_hidden]
        # mask = [batch, seq_len, seq_len]
        n_batch = query.shape[0] 

        # reshape [batch, seq_len, n_hidden] to [batch, n_head, seq_len, d_model]
        def transform(x, fc_layer):
            # x = [batch, seq_len, n_hidden]
            out = fc_layer(x) # n_hidden -> d_model, out = [batch, seq_len, d_model]
            out = out.view(n_batch, -1, self.n_head, self.n_hidden//self.n_head) 
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
        out = out.contiguous().view(n_batch, -1, self.n_hidden) # out = [batch, seq_len, n_hidden]
        out = self.fc_layer(out) # d_model -> d_embed. out = [batch, seq_len, d_embed]
        return out

    def calcuclate_attention(self, query, key, value, mask):
        d_k = key.size(-1) # query, key, value = [batch, n_head, seq_len, d_k]
        score = torch.matmul(query, key.transpose(-2,-1)) # Q x K^T
        score = score/ math.sqrt(d_k) # scailing        
        if mask is not None:
            # tensor.masked_fill(mask_index, value): Fills the elements with value where mask is True
            # score = score.masked_fill(mask == 0, -1e9) 
            score = score.masked_fill(mask == 1, -1e9) 
        out = F.softmax(score, dim = -1) # get the softmax score
        out = torch.matmul(out, value) # score x V
        return out

class Glimpse(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 n_head):
        super(Glimpse, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.single_dim = hidden_size // n_head
        self.c_div = 1.0 / math.sqrt(self.single_dim)
        # Why should we take bias?

        self.W_q = nn.Linear(self.input_size, self.hidden_size)
        self.W_k = nn.Linear(self.input_size, self.hidden_size)
        self.W_v = nn.Linear(self.input_size, self.hidden_size)
        self.W_out = nn.Linear(self.hidden_size, self.input_size)

        # No dropout or No Batch/Layernorm as mentioned at Wouter's paper

    def forward(self, query, target, mask=None):
        """
        [Inputs]
            query : FloatTensor with shape [batch x embedding_size]
            target : FloatTensor with shape [batch x seq_len x embedding_size]
            mask : BoolTensor with shape [batch_size x seq_len]
        [Outputs]
        """
        batch_size, seq_len, _ = target.shape

        q_c = self.W_q(query).reshape(batch_size, self.n_head, self.single_dim) # batch x head x head_dim

        k = self.W_k(target).reshape(batch_size, seq_len, self.n_head, self.single_dim).permute(0, 2, 1, 3).contiguous() # batch x head x seq_len x head_dim
        v = self.W_v(target).reshape(batch_size, seq_len, self.n_head, self.single_dim).permute(0, 2, 1, 3).contiguous() # batch x head x seq_len x head_dim
        qk = torch.einsum("ijl,ijkl->ijk", [q_c, k]) * self.c_div # batch x head x seq_len

        if mask is not None:
            _mask = mask.unsqueeze(1).repeat(1, self.n_head, 1) # batch x head x seq_len
            qk[_mask] = -100000.0

        alpha = torch.softmax(qk, -1) # batch x head x seq_len
        #print(alpha.shape, v.shape)
        h = torch.einsum("ijk,ijkl->ijl", alpha, v) # batch x head x head_dim

        if self.n_head == 1:
            ret = h.reshape(batch_size, -1)
            return alpha.squeeze(1), ret
        else:
            ret = self.W_out(h.reshape(batch_size, -1))
            return alpha, ret


class Multi_Layer(nn.Module):
    def __init__(self, n_heads, n_hidden, feed_forward_hidden = 512, bn=False):
        super(Multi_Layer, self).__init__()
        # self.mha = MultiHeadAttentionLayer(n_hidden= n_hidden, n_head = n_heads)
        # self.out = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_hidden))
        self.mha = torch.nn.MultiheadAttention(n_hidden, n_heads)
        self.out = nn.Sequential(nn.Linear(n_hidden, feed_forward_hidden), nn.ReLU(), nn.Linear(feed_forward_hidden, n_hidden))

    def forward(self, x):
        """"
        [input]
            x = batch X seq_len X embedding_size
        [output]
            _2 = batch X seq_len X embedding_size
        """
        x = x.permute(1,0,2)        
        _1 = x +  self.mha(x, x, x)[0]
        _1 = _1.permute(1, 0, 2)
        _2 = _1 + self.out(_1)
        return _2

class AttentionModule(nn.Sequential):
    def __init__(self, n_heads, n_hidden, feed_forward_hidden, n_layers = 3):
        super(AttentionModule, self).__init__(
            *(Multi_Layer(n_heads = n_heads, n_hidden=n_hidden, feed_forward_hidden= feed_forward_hidden) for _ in range(n_layers))
        )


