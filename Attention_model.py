import copy
import numpy as np
import torch
import torch.nn as nn
from Environment import Environment
from module import Pointer, MultiHeadAttentionLayer
from torch.distributions import Categorical


class Encoder(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, high_level):
        super(Encoder, self).__init__()
        cp = lambda x: copy.deepcopy(x)

        self.embedding_x = nn.Linear(n_feature, n_hidden)
        self.high_level = high_level
        self.multi_head_attention = MultiHeadAttentionLayer(
                                            d_model= n_hidden,
                                            n_head = 8,
                                            qkv_fc_layer= nn.Linear(n_hidden, n_hidden),
                                            fc_layer = nn.Linear(n_hidden, n_hidden))
        self.low_attention = cp(self.multi_head_attention)
        self.high_attention = cp(self.multi_head_attention)        

    def forward(self, batch_data, mask = None):
        self.batch_data = batch_data        
        self.low_node = []
        self.embedded_x = torch.tensor(())
        for batch_samples in self.batch_data:
            _x = torch.tensor(())
            _low_node = []
            for sample in batch_samples:
                x = self.embedding_x(sample.cuda()).unsqueeze(0)
                _low_node.append(self.low_attention(query = x, key = x, value = x, mask = None))                
            self.low_node.append(_low_node)                                
        
        # cell embedding
        self.high_node = self.cell_embedding(mask = None)

        return self.low_node, self.high_node, self.batch_data

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

class Decoder(torch.nn.Module):
    def __init__(self, n_embedding, seq_len, n_hidden, C = 10):
        super(Decoder, self).__init__()
        self.n_embedding = n_embedding
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.C = C
        
        self.high_pointer = Pointer(self.n_embedding, self.n_hidden, self.C)        
        self.init_w = nn.Parameter(torch.Tensor(2 * self.n_embedding))
        self.init_w.data.uniform_(-1,1)        
        self.h_context_embed = nn.Linear(self.n_embedding, self.n_embedding)
        self.v_weight_embed = nn.Linear(2 * self.n_embedding, self.n_embedding)

        self.init_h = None
        self.h = None
        # define low policy decoder
        self.low_decoder = Low_Decoder(n_embedding = self.n_embedding, n_hidden= self.n_hidden, C = 10)

    def forward(self, node_context, original_data, cell_context, high_mask, low_mask):
        self.cell_context = cell_context
        self.original_data = original_data
        self.batch_size = cell_context.size(0)        
        
        # calculate query
        h_mean = self.calculate_context(context_vector= self.cell_context) # batch * 1 * embedding_size
        h_bar = self.h_context_embed(h_mean) # batch * 1 * embedding_size
        h_rest = self.v_weight_embed(self.init_w) # 1 * embedding_size
        query = h_bar + h_rest # kool 논문에서는 concatenate 한다고 나와있지만 실제로는 아님...?

        # mask = [batch, n_cities]
        # query = [batch, embedding_size]
        temp_idx = []
        temp_log_idx = []

        # set the high environment
        self.high_environment = Environment(batch_data= self.cell_context, is_local= False)        
        high_mask[:, 0] = 1       
        
        for i in range(self.seq_len -1):                             
            # we could add glimpse later                        
            prob = self.high_pointer(query=query, target=self.cell_context, mask = high_mask)            
            node_distribtution = Categorical(prob)
            idx = node_distribtution.sample() # idx size = [batch]
            log_prob = node_distribtution.log_prob(idx) # change probability into log_prob
            
            # append sampled action into a list
            temp_idx.append(idx)
            temp_log_idx.append(log_prob)
                        
            # 현재 상태에서  low_level_policy를 실행해서 local CPP 알고리즘 실행
            low_log_prob = []
            low_action = []
            
            for id, (sub_node, sub_original_node, sub_mask) in enumerate(zip(node_context, self.original_data, low_mask)):
                low_index = 0 if i == 0 else idx.gather(0, torch.tensor(id).cuda())
                current_node = sub_node[low_index]
                original_node = sub_original_node[low_index]
                current_mask = sub_mask[low_index]                
                _low_log_prob, _low_action = self.low_decoder(current_node, original_node, current_mask)
                low_log_prob.append(_low_log_prob)
                low_action.append(_low_action)                
            # low_action size = [batch, local_seq_len, 1]                                         
            # low level policy에서 나온 log_prob 와 low_action이 어디로 들어가야하지?        

            # action masking for high level policy
            high_mask = self.high_environment.update_state(action_mask= high_mask, next_idx= idx)            
            
            _idx = idx.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.n_embedding) # torch.gather를 사용하기 위해 차원을 맞춰줌
            if self.init_h is None:
                self.init_h = self.cell_context.gather(1, _idx).squeeze(1)

            self.h = self.cell_context.gather(1, _idx).squeeze(1)
            h_rest = self.v_weight_embed(torch.cat((self.init_h, self.h), dim = -1)) # dim= -1 부분을 왜 이렇게 하는거지?
            query = h_bar + h_rest
                        
        return torch.stack(temp_log_idx, 1), torch.stack(temp_idx, 1)
                
    def calculate_context(self, context_vector):
        return torch.mean(context_vector, dim=1)

class Low_Decoder(torch.nn.Module):
    def __init__(self, n_embedding, n_hidden, C=10):
        super(Low_Decoder, self).__init__()
        self.n_embedding = n_embedding
        self.n_hidden = n_hidden
        self.C = C
        self.init_index = 0

        # initialize parameters
        self.low_init_w = nn.Parameter(torch.Tensor(2 * self.n_embedding))
        self.low_init_w.data.uniform_(-1,1)        
        self.low_h_context_embed = nn.Linear(self.n_embedding, self.n_embedding)
        self.low_v_weight_embed = nn.Linear(2 * self.n_embedding, self.n_embedding)

        # define pointer
        self.low_pointer = Pointer(self.n_embedding, self.n_hidden, self.C)

        # define inital indecies
        self.low_init_h = None
        self.low_h = None

    def forward(self, low_context_vector, original_node, mask):
        self.low_context_vector = low_context_vector
        self.original_node = original_node
        low_h_mean = self.calculate_low_context(low_context_vector=low_context_vector)
        low_h_bar = self.low_h_context_embed(low_h_mean)
        low_h_rest = self.low_v_weight_embed(self.low_init_w)
        low_query = low_h_bar + low_h_rest

        low_temp_idx = []
        low_temp_log_prob = []
        # mask size 잘 고려해서 넣자. low_mask는 dimension이 다르다.
        seq_len = low_context_vector.size(1)

        # set the environment
        self.low_environment = Environment(batch_data= self.low_context_vector, is_local= True)
        mask[self.init_index] = 1
        current_idx = torch.tensor([0]).cuda()

        for i in range(seq_len-1): # seq_len -1 에서 -1은 depot은 연산할 필요가 없기 때문!
            # change the initial mask 
                  
            low_prob = self.low_pointer(query = low_query, target = self.low_context_vector, mask = mask)            

            low_node_distribution = Categorical(low_prob)
            low_idx = low_node_distribution.sample()
            low_log_prob = low_node_distribution.log_prob(low_idx)
            
            # append the action and log_probability
            low_temp_idx.append(low_idx)
            low_temp_log_prob.append(low_log_prob)
                        
            # low mask action masking            
            mask = self.low_environment.update_state(action_mask= mask, next_idx = low_idx)
            next_idx = low_idx

            _low_idx = low_idx.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.n_embedding)
            if self.low_init_h is None:
                self.low_init_h = self.low_context_vector.gather(1, _low_idx).squeeze(1)
            
            self.low_h = self.low_context_vector.gather(1, _low_idx).squeeze(1)
            low_h_rest = self.low_v_weight_embed(torch.cat((self.low_init_h, self.low_h), dim = -1))
            low_query = low_h_bar + low_h_rest            

            # calculate reward
            local_reward = self.calculate_distance(self.original_node, current_index= current_idx, next_index= next_idx)
            current_idx = next_idx
            print("local reward:{}".format(local_reward))

        return torch.stack(low_temp_idx, 1), torch.stack(low_temp_log_prob, 1)

    def calculate_low_context(self, low_context_vector):
        return torch.mean(low_context_vector, dim = 1)

    def calculate_distance(self, local_nodes, current_index, next_index):        
        local_nodes = local_nodes.cuda()        
        current_node = local_nodes[current_index.data,:]   
        next_node = local_nodes[next_index.data,:]                              
        return torch.norm(next_node - current_node, dim=1)


class HCPP(torch.nn.Module):
    def __init__(self, 
                n_feature, 
                n_hidden, 
                high_level, 
                n_embedding, 
                seq_len, 
                C):
        super(HCPP, self).__init__()
        self.h_encoder = Encoder(n_feature= n_feature, n_hidden= n_hidden, high_level= high_level)
        self.h_decoder = Decoder(n_embedding= n_embedding, seq_len= seq_len, n_hidden= n_hidden, C = C)

    def forward(self, batch_data, high_mask, low_mask):            
        node_embed, cell_embed, original_data = self.h_encoder(batch_data, mask = None)        
        high_log_prob, high_action = self.h_decoder(node_embed, original_data, cell_embed, high_mask, low_mask)
        return high_log_prob, high_action


