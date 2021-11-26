import torch
import torch.nn as nn
from Environment import Environment
from module import Pointer, AttentionModule, Glimpse
from torch.distributions import Categorical


class Encoder(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_layer):
        super(Encoder, self).__init__()        
        self.n_hidden = n_hidden
        self.embedding_x = nn.Linear(n_feature, n_hidden)                
        
        self.low_attention = AttentionModule(
                                            n_heads = 8,
                                            n_hidden = n_hidden,
                                            feed_forward_hidden = 512,
                                            n_layers = n_layer
                                            )

        self.high_attention = AttentionModule(
                                            n_heads = 8,
                                            n_hidden = n_hidden,
                                            feed_forward_hidden = 512, 
                                            n_layers = n_layer
                                            )

    def forward(self, batch_data, mask = None):
        self.batch_data = batch_data.cuda()
        self.seq_len = self.batch_data.size(1)
        self.n_node = self.batch_data.size(2)

        # node embedding for low model
        self.embedded_x = self.embedding_x(self.batch_data)  
        _low_node = torch.reshape(self.embedded_x, [-1, self.seq_len, self.n_hidden])      
        _low_node = self.low_attention(_low_node)
        self.low_node = torch.reshape(_low_node, [-1, self.seq_len, self.n_node, self.n_hidden])
    
        # cell embedding for high model     
        # self.high_node = torch.mean(self.batch_data, dim = 2) 
        # self.high_node = self.embedding_x(self.high_node)
        """
        embedding 하고 나서 mean을 취하는것과 
        mean을 취하고 embedding 하는것의 차이는?
        """
        self.high_node = torch.mean(self.embedded_x, dim=2)
        self.high_node = self.high_attention(self.high_node)        
        return self.low_node, self.high_node, self.batch_data

class Decoder(torch.nn.Module):
    def __init__(self, n_embedding, n_hidden, n_head, C = 10):
        super(Decoder, self).__init__()
        self.n_embedding = n_embedding
        self.n_hidden = n_hidden        
        self.n_head = n_head
        self.C = C

        # define initial parameters
        self.init_w = nn.Parameter(torch.Tensor(2 * self.n_embedding))
        self.init_w.data.uniform_(-1,1)        
        self.h_context_embed = nn.Linear(self.n_embedding, self.n_embedding)
        self.v_weight_embed = nn.Linear(2 * self.n_embedding, self.n_embedding)

        # define pointer
        self.high_pointer = Pointer(self.n_embedding, self.n_hidden, self.C)        

    def forward(self, node_context, original_data, cell_context, high_mask, low_mask, low_decoder):        
        self.original_data = original_data
        self.batch_size = cell_context.size(0)        
        self.low_decoder = low_decoder
        self.seq_len = self.original_data.size(1)
        self.n_node = self.original_data.size(2)
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

            logits = self.high_pointer(query=query, target=cell_context, mask = high_mask)                   
            node_distribtution = Categorical(logits)
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
            current_cell = torch.gather(node_context, 1, idx.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.n_node, self.n_embedding))
            original_cell = torch.gather(self.original_data.cuda(), 1, idx.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.n_node, 2))
            current_mask = torch.gather(low_mask, 1, idx.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.n_node))
            
            # resize
            current_cell = current_cell.squeeze(1) # batch * n_seq_len * n_embedding
            original_cell = original_cell.squeeze(1) # batch * n_seq_len * n_feature
            current_mask = current_mask.squeeze(1) # batch * n_seq_len
                        
            for _ in range(self.seq_len):                
                # calculate low_log_prob and low_action
                _low_prob, _low_action, _reward, init_node, last_node  = self.low_decoder(current_cell, original_cell, current_mask, i)
                # append low elements into a list                                                
                node_log_prob.append(_low_prob)
                node_action.append(_low_action)
                node_reward.append(_reward)

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
        cell_reward = torch.sum(cell_reward.clone(), dim=-1)
        cell_log_prob = torch.sum(cell_log_prob.clone(), dim=-1)

        # resize the node_reward and node_log_prob
        "[batch, n_cells, local_cells] --> [batch*n_cells, local_cells]"
        node_reward = torch.reshape(node_reward, [-1, 1])
        node_log_prob = torch.reshape(node_log_prob, [-1, 1])   
    
        return cell_log_prob, node_log_prob, cell_reward, node_reward, cell_action, node_action
                
    def calculate_context(self, context_vector):
        return torch.mean(context_vector, dim=1)

    def calculate_cell_distance(self, init_node, last_node): 
        return torch.norm(last_node - init_node, dim=1).cuda()


class Low_Decoder(torch.nn.Module):
    def __init__(self, n_embedding, n_hidden, n_head, C=10):
        super(Low_Decoder, self).__init__()
        self.n_embedding = n_embedding
        self.n_hidden = n_hidden
        self.n_head = n_head
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
            low_node_distribution = Categorical(low_logits)
            low_idx = low_node_distribution.sample() # idx_size = [batch]
            
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

        # sum up log prob and reward
        low_temp_log_prob = torch.sum(low_temp_log_prob, dim=-1)          
        low_temp_R = torch.sum(low_temp_R, dim= -1)
        return low_temp_log_prob, low_temp_idx, low_temp_R, init_node, last_node 

    def calculate_low_context(self, low_context_vector):
        return torch.mean(low_context_vector, dim = 1)

    def calculate_distance(self, local_nodes, current_index, next_index):        
        local_nodes = local_nodes.cuda()              
        current_index = current_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, 2)
        next_index = next_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, 2)
        current_node = local_nodes.gather(1, current_index).squeeze(1)
        next_node = local_nodes.gather(1, next_index).squeeze(1)  
        return current_node, next_node, torch.norm(next_node - current_node, dim=1)

class HCPP(torch.nn.Module):
    def __init__(self, 
                n_feature, 
                n_hidden,                 
                n_embedding,                                 
                n_head,
                n_layer, 
                C):
        super(HCPP, self).__init__()
        self.h_encoder = Encoder(n_feature= n_feature, n_hidden= n_hidden, n_layer=n_layer)
        self.h_decoder = Decoder(n_embedding= n_embedding, n_hidden= n_hidden, n_head=n_head, C = C)        

    def forward(self, batch_data, high_mask, low_mask, low_decoder):     
        node_embed, cell_embed, original_data = self.h_encoder(batch_data, mask = None)                  
        high_log_prob, low_log_prob, high_reward, low_reward, high_action, low_action = self.h_decoder(node_embed, original_data, cell_embed, high_mask, low_mask, low_decoder)
        return high_log_prob, low_log_prob, high_reward, low_reward, high_action, low_action


