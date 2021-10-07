import torch
import torch.nn as nn
from Environment import Environment
from module import Pointer, MultiHeadAttentionLayer
from torch.distributions import Categorical


class Encoder(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, high_level):
        super(Encoder, self).__init__()        

        self.embedding_x = nn.Linear(n_feature, n_hidden)
        self.high_level = high_level
        self.multi_head_attention = MultiHeadAttentionLayer(
                                            n_hidden= n_hidden,
                                            n_head = 8)
        self.low_attention = MultiHeadAttentionLayer(
                                            n_hidden= n_hidden,
                                            n_head = 8)
        self.high_attention = MultiHeadAttentionLayer(
                                            n_hidden= n_hidden,
                                            n_head = 8)

    def forward(self, batch_data, mask = None):
        self.batch_data = batch_data
        self.low_node = []        
        for batch_samples in self.batch_data:            
            _low_node = []
            for sample in batch_samples:
                x = self.embedding_x(sample.clone().cuda()).unsqueeze(0) # x_size = [1, num_nodes, n_hidden]                
                _low_node.append(self.low_attention(query = x, key = x, value = x, mask = None))                
            self.low_node.append(_low_node)                                
        
        # cell embedding
        self.high_node = self.cell_embedding(mask = None) # high_node_size = [batch, n_cell, n_hidden]        
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
        
        # define low policy decoder
        self.low_decoder = Low_Decoder(n_embedding = self.n_embedding, n_hidden= self.n_hidden, C = 10)

    def forward(self, node_context, original_data, cell_context, high_mask, low_mask):        
        self.original_data = original_data
        self.batch_size = cell_context.size(0)        
        
        init_h = None
        h = None

        # calculate query
        h_mean = self.calculate_context(context_vector= cell_context) # batch * 1 * embedding_size
        h_bar = self.h_context_embed(h_mean) # batch * 1 * embedding_size
        h_rest = self.v_weight_embed(self.init_w) # 1 * embedding_size
        query = h_bar + h_rest # kool 논문에서는 concatenate 한다고 나와있지만 실제로는 아님...?

        # mask = [batch, n_cities]
        # query = [batch, embedding_size]        

        # set the high environment
        self.high_environment = Environment(batch_data= cell_context, is_local= False)    
        high_mask[:, 0] = 1   # n_cell의 가장 첫번째 부분만 1로 masking 작업 진행 // size: [batch, n_cells]
        
        # initialize node
        _last_node = torch.zeros([self.batch_size, 2]).cuda()  
        
        cell_log_prob = 0
        for i in range(self.seq_len):                                         
            # we could add glimpse later                                
            prob = self.high_pointer(query=query, target=cell_context, mask = high_mask)                   

            if not torch.isnan(prob)[0][0]:
                node_distribtution = Categorical(prob)
                idx = node_distribtution.sample() # idx size = [batch]                
                log_prob = node_distribtution.log_prob(idx) # change probability into log_prob
                                
                # calculate total log probability                
                cell_log_prob += torch.sum(log_prob, dim=-1)                            
                        
                # 현재 상태에서  low_level_policy를 실행해서 local CPP 알고리즘 실행
                low_log_prob = []                
                
                # define empty tensor to stack initial_node and last_node
                init_node = torch.tensor(()).cuda()
                last_node = torch.tensor(()).cuda()
                local_reward = 0
                local_log_prob = 0
                for id, (sub_node, sub_original_node, sub_mask) in enumerate(zip(node_context, self.original_data, low_mask)):                    
                    # get cell index
                    low_index = idx.gather(0, torch.tensor(id).cuda())

                    # get current node state
                    current_cell = sub_node[low_index].clone()
                    original_cell = sub_original_node[low_index].clone()
                    current_mask = sub_mask[low_index].clone()

                    # calculate low_log_prob and low_action
                    # init_node, last_node are nodes at current time step
                    _low_log_prob, _low_action, i_n, l_n, _local_reward = self.low_decoder(current_cell, original_cell, current_mask)
                    local_log_prob += torch.sum(_low_log_prob, dim=1)

                    # append log_prob and action                
                    low_log_prob.append(_low_log_prob)                    

                    # stack the local reward / local_reward size = []
                    local_reward += _local_reward
                    # stack the last nodes                    
                    init_node = torch.cat((init_node, i_n), dim=0) # _init_node size: [batch, 2]
                    last_node = torch.cat((last_node, l_n), dim=0) # _last_node size: [batch, 2]                
                                           

                # calculate cell distance                
                cell_reward = self.calculate_cell_distance(init_node= init_node, last_node=_last_node)                
                cell_reward = torch.sum(cell_reward, dim=0)                

                # update the node states
                _last_node = last_node.clone()                
                    
                # action masking for high level policy
                high_mask = self.high_environment.update_state(action_mask= high_mask, next_idx= idx)            
                
                _idx = idx.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.n_embedding) # torch.gather를 사용하기 위해 차원을 맞춰줌
                if init_h is None:
                    init_h = cell_context.gather(1, _idx).squeeze(1)

                h = cell_context.gather(1, _idx).squeeze(1)
                h_rest = self.v_weight_embed(torch.cat([init_h, h], dim = -1)) # dim= -1 부분을 왜 이렇게 하는거지?
                query = h_bar + h_rest
                total_reward = cell_reward + local_reward                  
                total_log_prob = cell_log_prob + local_log_prob

        return total_log_prob, total_reward
                
    def calculate_context(self, context_vector):
        return torch.mean(context_vector, dim=1)

    def calculate_cell_distance(self, init_node, last_node):
        return torch.norm(last_node - init_node, dim=1).cuda()


class Low_Decoder(torch.nn.Module):
    def __init__(self, n_embedding, n_hidden, C=10):
        super(Low_Decoder, self).__init__()
        self.n_embedding = n_embedding
        self.n_hidden = n_hidden
        self.C = C        

        # initialize parameters
        self.low_init_w = nn.Parameter(torch.Tensor(2 * self.n_embedding))
        self.low_init_w.data.uniform_(-1,1)        
        self.low_h_context_embed = nn.Linear(self.n_embedding, self.n_embedding)
        self.low_v_weight_embed = nn.Linear(2 * self.n_embedding, self.n_embedding)

        # define pointer
        self.low_pointer = Pointer(self.n_embedding, self.n_hidden, self.C)

    def forward(self, low_context_vector, original_node, mask):        
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
        self.low_environment = Environment(batch_data= low_context_vector, is_local= True)        
        current_idx = torch.tensor([0]).cuda()

        # define inital indecies
        low_init_h = None
        low_h = None

        # initialize        
        init_node = None
        last_node = None
        local_R = 0
        for i in range(seq_len): 
            # change the initial mask                  
            low_prob = self.low_pointer(query = low_query, target = low_context_vector, mask = mask)                     
            
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
            if low_init_h is None:
                low_init_h = low_context_vector.gather(1, _low_idx).squeeze(1)
            
            low_h = low_context_vector.gather(1, _low_idx).squeeze(1)
            low_h_rest = self.low_v_weight_embed(torch.cat((low_init_h, low_h), dim = -1))
            low_query = low_h_bar + low_h_rest            

            # calculate reward
            _current_node, _next_node, local_reward = self.calculate_distance(self.original_node, current_index= current_idx, next_index= next_idx) 
            current_idx = next_idx.clone()                                   
            
            # sum the loca reward
            local_R +=local_reward.clone()            

            # get the node state to calculate the distance between current cell and next cell
            if i == 0:
                init_node = _current_node.clone()
            if i == seq_len-1:
                last_node = _next_node.clone()            
        
        return torch.stack(low_temp_log_prob, 1), torch.stack(low_temp_idx, 1), init_node, last_node, local_R

    def calculate_low_context(self, low_context_vector):
        return torch.mean(low_context_vector, dim = 1)

    def calculate_distance(self, local_nodes, current_index, next_index):        
        local_nodes = local_nodes.cuda()                
        current_node = local_nodes[current_index.data,:]   
        next_node = local_nodes[next_index.data,:]                              
        return current_node, next_node, torch.norm(next_node - current_node, dim=1)


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
        log_prob, reward = self.h_decoder(node_embed, original_data, cell_embed, high_mask, low_mask)
        return log_prob, reward


