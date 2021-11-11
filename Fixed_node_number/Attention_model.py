import torch
import torch.nn as nn
from Environment import Environment
from module import Pointer, AttentionModule
from torch.distributions import Categorical


class Encoder(torch.nn.Module):
    def __init__(self, n_feature, n_hidden):
        super(Encoder, self).__init__()        

        self.embedding_x = nn.Linear(n_feature, n_hidden)                
        
        # self.low_attention = AttentionModule(
        #                                     n_heads = 8,
        #                                     n_hidden = n_hidden,
        #                                     n_layers = 3
        #                                     )

        self.high_attention = AttentionModule(
                                            n_heads = 8,
                                            n_hidden = n_hidden,
                                            n_layers = 3
                                            )

    def forward(self, batch_data, mask = None):
        self.batch_data = batch_data.cuda()
        self.low_node = self.embedding_x(self.batch_data)
        
        # self.low_node = []
        # for samples in batch_data:
        #     _low_node = []
        #     for sample in samples:                
        #         x = self.embedding_x(sample.clone().cuda()).unsqueeze(0) # x_size = [1, num_nodes, n_hidden]                   
        #         _low_node.append(self.low_attention(x))
        #     _low_node = torch.stack(_low_node, 1)                       
        #     self.low_node.append(_low_node.squeeze(0))            
        # self.low_node = torch.stack(self.low_node, 1)
        # self.low_node = torch.permute(self.low_node, (1, 2, 0 ,3))
        # # self.low_node = torch.permute(self.low_node, (1, 0, 2 ,3))

        # cell embedding for high model        
        self.high_node = self.embedding_x(self.batch_data)
        self.high_node = torch.mean(self.high_node, dim=2)
        self.high_node = self.high_attention(self.high_node)
        # print(self.high_node.size())

        return self.low_node, self.high_node, self.batch_data

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

        # set the high environment
        self.high_environment = Environment(batch_data= cell_context, is_local= False)            
        
        # initialize node
        _last_node = torch.zeros([self.batch_size, 2]).cuda()  
        
        # initialize cell log_prob and cell reward                
        cell_log_prob = []
        cell_reward = []        
        cell_action = []

        # depot을 0으로 바꾸면서 masking의 갯수가 문제가 생겼음... 
        for i in range(self.seq_len):         
            # initialize node elements
            node_log_prob = []
            node_reward = []
            node_action = []

            # we could add glimpse later                                
            logits = self.high_pointer(query=query, target=cell_context, mask = high_mask)       
            _high_mask = high_mask.clone()
            logits = torch.masked_fill(logits, _high_mask==1, float('-inf'))                                    
            prob = torch.softmax(logits, dim=-1)
            node_distribtution = Categorical(prob)
            idx = node_distribtution.sample() # idx size = [batch]                        
            # for the home cell
            if i==0:
                idx = torch.zeros([self.batch_size,], dtype = torch.int64).cuda()                
            _cell_log_prob = node_distribtution.log_prob(idx)
            # append high action
            cell_action.append(idx)

            # calculate total log probability
            cell_log_prob.append(_cell_log_prob)                    
                                                                                    
            """
            For Low Model
            """
            current_cell = torch.gather(node_context, 1, idx.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.seq_len, self.n_embedding))
            original_cell = torch.gather(self.original_data.cuda(), 1, idx.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.seq_len, 2))
            current_mask = torch.gather(low_mask, 1, idx.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.seq_len))
            
            # resize
            current_cell = current_cell.squeeze(1) # batch * n_seq_len * n_embedding
            original_cell = original_cell.squeeze(1) # batch * n_seq_len * n_feature
            current_mask = current_mask.squeeze(1) # batch * n_seq_len
                        
            for _ in range(self.seq_len):                
                # calculate low_log_prob and low_action
                _low_prob, _low_action, init_node, last_node, _reward = self.low_decoder(current_cell, original_cell, current_mask, i)
                      
                # append low elements into a list                                                
                node_log_prob.append(_low_prob)
                node_action.append(_low_action)
                node_reward.append(_reward)

            # reshape the init_node and last node
            """
            init_node: batch * 1 * n_feature --> batch * n_feature
            last_node: batch * 1 * n_feature --> batch * n_feature
            """
            init_node = init_node.squeeze(1)
            last_node = last_node.squeeze(1)

            if i > 0:        
                # calculate cell distance    
                _cell_reward = self.calculate_cell_distance(init_node= init_node, last_node=_last_node)
                # append cell_reward
                cell_reward.append(_cell_reward)   
            
            # update the last node                       
            _last_node = last_node.clone()

            # action masking for high level policy
            high_mask = self.high_environment.update_state(action_mask= high_mask, next_idx= idx)            
            
            _idx = idx.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.n_embedding) # torch.gather를 사용하기 위해 차원을 맞춰줌
            if init_h is None:
                init_h = cell_context.gather(1, _idx).squeeze(1)

            h = cell_context.gather(1, _idx).squeeze(1)
            h_rest = self.v_weight_embed(torch.cat([init_h, h], dim = -1)) # dim= -1 부분을 왜 이렇게 하는거지?
            query = h_bar + h_rest   
        
        
        
        # reshape the dimensions
        """
        cell_reward = batch * (n_cells - 1)
        cell_log_prob = batch * n_cells
        cell_action = batch * n_cells

        node_log_prob: batch * n_cells * local_cells
        node_action: batch * n_cells * local_cells
        node_reward: batch * n_cells * local_cells
        """        
        cell_log_prob = torch.stack(cell_log_prob, 1)
        cell_action = torch.stack(cell_action, 1) 
        cell_reward = torch.stack(cell_reward, 1)

        node_log_prob = torch.stack(node_log_prob, 1)
        node_action = torch.stack(node_action, 1)
        node_reward = torch.stack(node_reward, 1)
        
        # sum up log probability and reward                 
        """
        cell_reward = batch  
        cell_log_prob = batch
        """        
        cell_reward = torch.sum(cell_reward.clone(), dim=1)
        cell_log_prob = torch.sum(cell_log_prob.clone(), dim=1)

        # resize the node_reward and node_log_prob
        "[batch, n_cells, local_cells] --> [batch*n_cells, local_cells]"
        node_reward = torch.reshape(node_reward, [-1, self.seq_len])
        node_log_prob = torch.reshape(node_log_prob, [-1 , self.seq_len])
        node_reward = torch.sum(node_reward.clone(), dim=1)
        node_log_prob = torch.sum(node_log_prob.clone(), dim=1)

        return cell_log_prob, node_log_prob, cell_reward, node_reward, cell_action, node_action
                
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
        self.batch_size = low_context_vector.size(0)
        low_cv = low_context_vector.clone()
        self.original_node = original_node        
        
        low_h_mean = self.calculate_low_context(low_context_vector=low_cv)
        low_h_bar = self.low_h_context_embed(low_h_mean)
        low_h_rest = self.low_v_weight_embed(self.low_init_w)
        low_query = low_h_bar + low_h_rest

        low_temp_idx = []
        low_temp_log_prob = []
        low_temp_R = []

        # mask size 잘 고려해서 넣자. low_mask는 dimension이 다르다.
        seq_len = low_context_vector.size(1)

        # set the environment
        self.low_environment = Environment(batch_data= low_cv, is_local= True)        
        current_idx = torch.zeros([self.batch_size], dtype= torch.int64).cuda()        

        # define inital indecies
        low_init_h = None
        low_h = None

        # initialize        
        init_node = None
        last_node = None

        for i in range(seq_len): 
            low_logits = self.low_pointer(query = low_query, target = low_cv, mask = mask)   
            _mask = mask.clone()
            low_logits = torch.masked_fill(low_logits, _mask==1, float('-inf'))                           
            low_prob = torch.softmax(low_logits, dim=-1)                       
            low_node_distribution = Categorical(low_prob)
            low_idx = low_node_distribution.sample()   
            
            # change the initial mask              
            if i == 0 and id ==0:                      
                low_idx = torch.zeros([self.batch_size,], dtype = torch.int64).cuda()
            _low_log_prob = low_node_distribution.log_prob(low_idx)            

            # append the action and log_probability
            low_temp_idx.append(low_idx)
            low_temp_log_prob.append(_low_log_prob)
            
            # low mask action masking            
            """
            mask size: [batch, n_cells]
            low_idx size: [batch, ]
            """            
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
            
            # update the current index            
            current_idx = next_idx.clone()
            
            # append the reward
            low_temp_R.append(local_reward)            

            # get the node state to calculate the distance between current cell and next cell
            if i == 0:
                init_node = _current_node.clone()
            if i == seq_len-1:
                last_node = _next_node.clone()   

        low_temp_idx = torch.stack(low_temp_idx, 1)
        low_temp_log_prob = torch.stack(low_temp_log_prob, 1)     
        low_temp_R = torch.stack(low_temp_R, 1)          
        low_temp_R = torch.sum(low_temp_R, dim= 2)
        return low_temp_log_prob, low_temp_idx, init_node, last_node, low_temp_R

    def calculate_low_context(self, low_context_vector):
        return torch.mean(low_context_vector, dim = 1)

    def calculate_distance(self, local_nodes, current_index, next_index):        
        local_nodes = local_nodes.cuda()              
        current_index = current_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, 2)
        next_index = next_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, 2)
        current_node = torch.gather(local_nodes, 1, current_index)        
        next_node = torch.gather(local_nodes, 1, next_index)

        return current_node, next_node, torch.norm(next_node - current_node, dim=2)

class HCPP(torch.nn.Module):
    def __init__(self, 
                n_feature, 
                n_hidden,                 
                n_embedding, 
                seq_len,                 
                C):
        super(HCPP, self).__init__()
        self.h_encoder = Encoder(n_feature= n_feature, n_hidden= n_hidden)
        self.h_decoder = Decoder(n_embedding= n_embedding, seq_len= seq_len, n_hidden= n_hidden, C = C)        

    def forward(self, batch_data, high_mask, low_mask, low_decoder):     
        node_embed, cell_embed, original_data = self.h_encoder(batch_data, mask = None)                  
        high_log_prob, low_log_prob, high_reward, low_reward, high_action, low_action = self.h_decoder(node_embed, original_data, cell_embed, high_mask, low_mask, low_decoder)
        return high_log_prob, low_log_prob, high_reward, low_reward, high_action, low_action


