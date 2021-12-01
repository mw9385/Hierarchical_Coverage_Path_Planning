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
        self.n_cell = self.batch_data.size(1)
        self.n_node = self.batch_data.size(2)        
        self.low_node = []
        # node embedding for low model
        self.embedded_x = self.embedding_x(self.batch_data)
        
        for i in range(self.n_cell):
            temp_node_embed = self.embedded_x[:, i, :, :] # [batch_size, n_node, n_embedding]
            _low_node = self.low_attention(temp_node_embed) # [batch_size, n_node, n_embedding]
            self.low_node.append(_low_node)            
        self.low_node = torch.stack(self.low_node, 1)

        # embedding and attention for high model
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

    def forward(self, node_context, cell_context, original_data, high_mask, low_mask, low_decoder):                
        self.batch_size = original_data.size(0)
        self.n_cell = original_data.size(1)
        self.n_node = original_data.size(2)
        
        self.original_data = original_data
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

        # initialize node elements
        node_log_prob = []
        node_reward = []
        node_action = []

        for i in range(self.n_cell):                     
            logits = self.high_pointer(query=query, target=cell_context, mask = high_mask)                   
            node_distribtution = Categorical(logits)
            idx = node_distribtution.sample() # idx size = [batch]                        
            _cell_log_prob = node_distribtution.log_prob(idx)
            
            # append high action
            cell_action.append(idx)
            # calculate total log probability
            cell_log_prob.append(_cell_log_prob)                    
                                                                                    
            # get current cell information
            current_cell = torch.gather(node_context, 1, idx.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.n_node, self.n_embedding))
            original_cell = torch.gather(self.original_data.cuda(), 1, idx.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.n_node, 2))
            current_mask = low_mask.clone()            

            # resize
            current_cell = current_cell.squeeze(1) # batch * n_seq_len * n_embedding
            original_cell = original_cell.squeeze(1) # batch * n_seq_len * n_feature                                    
            
            # produce local routes using low decoder
            _low_prob, _low_action, _reward, init_node, last_node = self.low_decoder(current_cell, original_cell, current_mask)
            # store node information in lists
            """
            node_log_prob = [batch_size, 1]
            node_action = [batch_size, n_node]
            node_reward = [batch_size, 1]
            """
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
        cell_log_prob = batch * n_cells        
        cell_action = batch * n_cells
        cell_reward = batch * (n_cells - 1)        

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

        # resize the node elements
        "[batch, n_cells] --> [batch*n_cells,1]"
        node_log_prob = torch.reshape(node_log_prob, [-1, 1])        
        node_action = torch.reshape(node_action, [-1, self.n_node])
        node_reward = torch.reshape(node_reward, [-1, 1])

        # sum up log probability and reward                                 
        cell_reward = torch.sum(cell_reward.clone(), dim=-1)
        cell_log_prob = torch.sum(cell_log_prob.clone(), dim=-1)                
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

    def forward(self, low_context_vector, original_node, mask): 
        # get the dimension
        self.batch_size = low_context_vector.size(0)        
        self.n_node = low_context_vector.size(1)
        self.original_node = original_node        
                
        # calculate query 
        low_h_mean = self.calculate_low_context(low_context_vector=low_context_vector)
        low_h_bar = self.low_h_context_embed(low_h_mean)
        low_h_rest = self.low_v_weight_embed(self.low_init_w)
        low_query = low_h_bar + low_h_rest
        
        # initialize node_log_prob and node_reward
        low_temp_idx = []
        low_temp_log_prob = []
        low_temp_R = []
            
        # set the environment
        self.low_environment = Environment(batch_data= low_context_vector, is_local= False)                      

        # define inital indecies
        low_init_h = None
        low_h = None        

        #initialize the node
        self.init_node = torch.zeros([self.batch_size, 2]).cuda()
        self.last_node = torch.zeros([self.batch_size, 2]).cuda()        

        for i in range(self.n_node):             
            low_logits = self.low_pointer(query = low_query, target = low_context_vector, mask = mask)               
            low_node_distribution = Categorical(low_logits)
            low_idx = low_node_distribution.sample() # idx_size = [batch]
            _low_log_prob = low_node_distribution.log_prob(low_idx)            
            
            # append the action and log_probability
            low_temp_idx.append(low_idx)
            low_temp_log_prob.append(_low_log_prob)
                        
            mask = self.low_environment.update_state(action_mask= mask, next_idx = low_idx)            
            
            # get the current node
            self.init_node = self.original_node.gather(1, low_idx.unsqueeze(1).unsqueeze(2).repeat(1, 1, 2))            
            # calculate reward
            if i > 0:
                local_reward = self.calculate_distance(init_node = self.init_node.squeeze(1), last_node = self.last_node.squeeze(1))                 
                # append the reward
                low_temp_R.append(local_reward)            

            _low_idx = low_idx.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.n_embedding)
            if low_init_h is None:
                low_init_h = low_context_vector.gather(1, _low_idx).squeeze(1)
            
            low_h = low_context_vector.gather(1, _low_idx).squeeze(1)
            low_h_rest = self.low_v_weight_embed(torch.cat((low_init_h, low_h), dim = -1))
            low_query = low_h_bar + low_h_rest            

            # get current node
            self.last_node = self.init_node.clone()
                        
            # get the node state to calculate the distance between current cell and next cell
            if i == 0:
                start_node = self.original_node.gather(1,low_idx.unsqueeze(1).unsqueeze(2).repeat(1,1,2))
                start_node = start_node.squeeze(1)                
            if i == self.n_node-1:
                end_node = self.original_node.gather(1,low_idx.unsqueeze(1).unsqueeze(2).repeat(1,1,2))
                end_node = end_node.squeeze(1)

        low_temp_idx = torch.stack(low_temp_idx, 1)
        low_temp_log_prob = torch.stack(low_temp_log_prob, 1) 
        low_temp_R = torch.stack(low_temp_R, 1)        
        
        # sum up log prob and reward
        low_temp_R = torch.sum(low_temp_R, dim= -1)
        low_temp_log_prob = torch.sum(low_temp_log_prob, dim=-1)    
        return low_temp_log_prob, low_temp_idx, low_temp_R, start_node, end_node 

    def calculate_low_context(self, low_context_vector):
        return torch.mean(low_context_vector, dim=1)

    def calculate_distance(self, init_node, last_node): 
        return torch.norm(last_node - init_node, dim=1).cuda()


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
        high_log_prob, low_log_prob, high_reward, low_reward, high_action, low_action = self.h_decoder(node_embed, cell_embed, original_data, high_mask, low_mask, low_decoder)
        return high_log_prob, low_log_prob, high_reward, low_reward, high_action, low_action


