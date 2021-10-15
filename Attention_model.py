import torch
import torch.nn as nn
from Environment import Environment
from module import Pointer, MultiHeadAttentionLayer, AttentionModule
from torch.distributions import Categorical


class Encoder(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, high_level):
        super(Encoder, self).__init__()        

        self.embedding_x = nn.Linear(n_feature, n_hidden)
        self.high_level = high_level                
        self.low_attention = AttentionModule(
                                            n_heads = 8,
                                            n_hidden = n_hidden,
                                            n_layers = 3
                                            )
        self.high_attention = AttentionModule(
                                            n_heads = 8,
                                            n_hidden = n_hidden,
                                            n_layers = 3
                                            )

    def forward(self, batch_data, mask = None):
        self.batch_data = batch_data
        self.low_node = []        
        for batch_samples in self.batch_data:            
            _low_node = []
            for sample in batch_samples:
                x = self.embedding_x(sample.clone().cuda()).unsqueeze(0) # x_size = [1, num_nodes, n_hidden] 
                _low_node.append(self.low_attention(x))                                               
                # _low_node.append(self.low_attention(query = x, key = x, value = x, mask = None))                
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
        att_cell = self.high_attention(embedded_mean)
        # att_cell = self.high_attention(query = embedded_mean, key = embedded_mean, value = embedded_mean, mask = None)
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
        
    def forward(self, node_context, original_data, cell_context, high_mask, low_mask, low_decoder):        
        self.original_data = original_data
        self.batch_size = cell_context.size(0)        
        self.low_decoder = low_decoder
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
        
        # initialize node
        _last_node = torch.zeros([self.batch_size, 2]).cuda()  
        
        # initialize cell log_prob and cell reward
        # list 형태를 사용하지 않고 torch cat을 사용하려고 했었는데, loop 안에서 반복적으로 들어가게 되면서 inplace operation error 발생
        # 차선책으로 list로 tensor를 쌓아두고, torch.stack을 사용해서 list를 tensor 형태로 사용하는 방법을 이용함.
        cell_log_prob = []
        cell_reward = []
        node_log_prob = []
        node_reward = []
        low_action = []
        high_action = []

        # depot을 0으로 바꾸면서 masking의 갯수가 문제가 생겼음... 
        for i in range(self.seq_len):            
            # we could add glimpse later                                
            prob = self.high_pointer(query=query, target=cell_context, mask = high_mask)                   
            node_distribtution = Categorical(prob)
            idx = node_distribtution.sample() # idx size = [batch]            
            
            # for the home cell
            if i==0:
                idx = torch.zeros([self.batch_size,], dtype = torch.int64).cuda()                
            _cell_log_prob = node_distribtution.log_prob(idx)

            # append high action
            high_action.append(idx)

            # calculate total log probability
            cell_log_prob.append(_cell_log_prob)                    
                                        
            # 현재 상태에서  low_level_policy를 실행해서 local CPP 알고리즘 실행
            _low_log_prob = []
            _low_reward = []
            _low_action = []
                
            # define empty tensor to stack initial_node and last_node
            init_node = torch.tensor(()).cuda()
            last_node = torch.tensor(()).cuda()   
                
            for id, (sub_node, sub_original_node, sub_mask) in enumerate(zip(node_context, self.original_data, low_mask)):                                        
                # batch size 개수만큼 for loop를 돌린다. 
                # get cell index
                low_index = idx.gather(0, torch.tensor(id).cuda())                                    
                # get current node state
                current_cell = sub_node[low_index].clone()
                original_cell = sub_original_node[low_index].clone()
                current_mask = sub_mask[low_index].clone()                

                # calculate low_log_prob and low_action
                # init_node, last_node are nodes at current time step                
                _log_prob, _l_a, i_n, l_n, _reward = self.low_decoder(current_cell, original_cell, current_mask, i)                                                        
                _sum_p = torch.sum(_log_prob, dim=0)                    
                
                # append log_prob and action     
                _low_log_prob.append(_sum_p)
                _low_reward.append(_reward.squeeze(0))                                    
                _low_action.append(_l_a.squeeze(0)) # [batch, nodes]                
                
                # stack the last nodes                    
                init_node = torch.cat((init_node, i_n), dim=0) # _init_node size: [batch, 2]
                last_node = torch.cat((last_node, l_n), dim=0) # _last_node size: [batch, 2]                                
            
            _low_log_prob = torch.stack(_low_log_prob, 0)
            _low_reward = torch.stack(_low_reward, 0)

            # _low_action은 list 형태이기 때문에 torch.stack을 사용하지 못함. _low_action = [batch size, # number of local nodes]
            # stack inner log prob and reward
            node_log_prob.append(_low_log_prob)
            node_reward.append(_low_reward)                          
            low_action.append(_low_action) # list shape = [cell, batch, # nodes]
            
            # calculate cell distance    
            _cell_reward = self.calculate_cell_distance(init_node= init_node, last_node=_last_node)
            cell_reward.append(_cell_reward)                            
            # cell_reward.append(self.calculate_cell_distance(init_node= init_node, last_node=_last_node))                

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
        
        # change the dimension of reward and log_probability
        # size of cell_reward, cell_log_prob, low_reward, low_log_prob = [batch]           
        high_reward = torch.stack(cell_reward, 1)
        high_log_prob = torch.stack(cell_log_prob, 1)
        low_reward = torch.stack(node_reward, 1)
        low_log_prob = torch.stack(node_log_prob, 1)
        high_action = torch.stack(high_action, 1) # batch x n_cells -1
        #low_action size = [n_cells-1, batch, # of nodes]

        high_reward = torch.sum(high_reward.clone(), dim=1)
        high_log_prob = torch.sum(high_log_prob.clone(), dim=1)
        low_reward = torch.sum(low_reward.clone(), dim=1)
        low_log_prob = torch.sum(low_log_prob.clone(), dim=1)

        # aa_index = 0
        # sample = original_data[aa_index]        
        # bb = high_action[aa_index]        
        # print(sample)
        # print(bb)
        # print('----------------')
        # for ss in low_action:            
        #     print(ss[aa_index])

        return high_log_prob, low_log_prob, high_reward, low_reward, high_action, low_action
                
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

    def forward(self, low_context_vector, original_node, mask, id): 
        low_cv = low_context_vector.clone()
        self.original_node = original_node        
        
        low_h_mean = self.calculate_low_context(low_context_vector=low_cv)
        low_h_bar = self.low_h_context_embed(low_h_mean)
        low_h_rest = self.low_v_weight_embed(self.low_init_w)
        low_query = low_h_bar + low_h_rest

        low_temp_idx = []
        low_temp_log_prob = torch.tensor(()).cuda()
        # mask size 잘 고려해서 넣자. low_mask는 dimension이 다르다.
        seq_len = low_context_vector.size(1)

        # set the environment
        self.low_environment = Environment(batch_data= low_cv, is_local= True)        
        current_idx = torch.tensor([0]).cuda()

        # define inital indecies
        low_init_h = None
        low_h = None

        # initialize        
        init_node = None
        last_node = None
        local_R = 0
        for i in range(seq_len): 
            low_prob = self.low_pointer(query = low_query, target = low_cv, mask = mask)                     
            low_node_distribution = Categorical(low_prob)
            low_idx = low_node_distribution.sample()            
            # change the initial mask               
            if i == 0 and id ==0:
                low_idx = torch.zeros([1,], dtype=torch.int64).cuda()            
            _low_log_prob = low_node_distribution.log_prob(low_idx)            
            # print(_low_log_prob)

            # append the action and log_probability
            low_temp_idx.append(low_idx)
            low_temp_log_prob = torch.cat((low_temp_log_prob, _low_log_prob.clone()), dim=0)            
                        
            # low mask action masking            
            mask = self.low_environment.update_state(action_mask= mask, next_idx = low_idx)
            next_idx = low_idx.clone()

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
        return low_temp_log_prob.clone(), torch.stack(low_temp_idx, 1), init_node, last_node, local_R

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

    def forward(self, batch_data, high_mask, low_mask, low_decoder):     
        node_embed, cell_embed, original_data = self.h_encoder(batch_data, mask = None)                  
        high_log_prob, low_log_prob, high_reward, low_reward, high_action, low_action = self.h_decoder(node_embed, original_data, cell_embed, high_mask, low_mask, low_decoder)
        return high_log_prob, low_log_prob, high_reward, low_reward, high_action, low_action


