from numpy import dtype
import torch
import torch.nn as nn
from Environment import Environment
from module import Pointer, AttentionModule, Glimpse
from torch.distributions import Categorical
from A_star import *

class Encoder(torch.nn.Module):
    def __init__(self, n_feature, n_hidden):
        super(Encoder, self).__init__()        
        self.n_hidden = n_hidden
        self.n_feature = n_feature
        self.max_length = 100
        self.embedding_x = nn.Linear(n_feature, n_hidden)                        
        self.high_node = []
        self.high_attention = AttentionModule(
                                            n_heads = 8,
                                            n_hidden = n_hidden,
                                            feed_forward_hidden= 512,
                                            n_layers = 3
                                            )

    def forward(self, batch_data, costs, mask = None):    
        # cell embedding for high model  
        self.n_batch = len(batch_data)      
        self.high_node = torch.zeros([self.n_batch, self.max_length, self.n_hidden], dtype=torch.float32).cuda()
        self.original_node = torch.zeros([self.n_batch, self.max_length, self.n_feature], dtype=torch.float32).cuda()
        self.costs = torch.zeros([self.n_batch, self.max_length], dtype= torch.float32).cuda() * 10000        
        for index, batch_sample in enumerate(batch_data): 
            # data normalization
            A,B,C,D = batch_sample.size()
            batch_sample = batch_sample.reshape([A, B, C * D]).cuda()
            high_node = self.embedding_x(batch_sample.type(torch.float32).cuda() / 70.0) 
            high_node = self.high_attention(high_node) # size = [4 * n_cells, n_feature, n_hidden]             
            self.high_node[index, :B, :] = high_node
            self.original_node[index, :B, :] = batch_sample
            self.costs[index, :B] = costs[index]
        return self.high_node, self.original_node, self.costs

# define decoder
class Decoder(torch.nn.Module):
    def __init__(self, n_embedding, n_hidden, n_feature, n_head, C = 10):
        super(Decoder, self).__init__()
        self.n_embedding = n_embedding
        self.n_hidden = n_hidden        
        self.n_head = n_head
        self.C = C
        self.max_length = 100

        # define initial parameters
        self.init_w = nn.Parameter(torch.Tensor(2 * self.n_embedding))
        self.init_w.data.uniform_(-1,1)        
        self.h_context_embed = nn.Linear(self.n_embedding, self.n_embedding)
        self.v_weight_embed = nn.Linear(2 * self.n_embedding, self.n_embedding)

        # define pointer
        self.high_pointer = Pointer(self.n_embedding, self.n_hidden, self.C)                        
        # encoder
        self.embedding_x = nn.Linear(n_feature, n_hidden)                        
        self.high_attention = AttentionModule(
                                            n_heads = 8,
                                            n_hidden = n_hidden,
                                            feed_forward_hidden= 512,
                                            n_layers = 3
                                            )

    def forward(self, high_node, original_node, map, num_cell, costs):
        self.n_batch = high_node.size(0)
        init_h = None
        h = None
        # do action masking if the element of the inputs are zeros and find the element with zeros
        high_mask = torch.zeros([self.n_batch, self.max_length], dtype= torch.int64).cuda() 
        non_zero_idx = torch.where(high_node==0, 1, 0)    
        high_mask = torch.masked_fill(high_mask, non_zero_idx[:,:,0]==1, 1)
        high_mask[:, :4] = 1

        # calculate query
        h_mean = self.calculate_mean(high_node) # batch * 1 * embedding_size
        h_bar = self.h_context_embed(h_mean) # batch * 1 * embedding_size
        h_rest = self.v_weight_embed(self.init_w) # 1 * embeddig_size            
        query = h_bar + h_rest # batch * embedding_size        
        # set the high environment
        self.high_environment = Environment(batch_data = high_node)    

        # initialize cell log_prob and cell reward
        cell_log_prob, cell_reward, cell_action = [], [], []
        
        for i in range(self.max_length):
            # calculate logits and node index
            logits = self.high_pointer(query=query, target=high_node, mask = high_mask)    
            node_distribtution = Categorical(logits)
            idx = node_distribtution.sample()
            if i == 0:
                idx = torch.argmin(costs[:, :4], dim=1)                     
            _cell_log_prob = node_distribtution.log_prob(idx)

            # change the log probability into 0 if summed mask is equal to sequence length            
            done_index = torch.where(num_cell<=i, 1, 0).cuda().squeeze(1)
            _cell_log_prob = torch.masked_fill(_cell_log_prob, done_index == 1, 0)            
            # append the log probabiltiy and actions
            cell_log_prob.append(_cell_log_prob)
            cell_action.append(idx)

            # action maksing for policy
            I = (idx / 4).type(torch.int64).unsqueeze(1) * 4
            for _ in range(4):
                high_mask = high_mask.scatter(1, I.data, 1).clone()                 
                I = I.clone() + torch.ones([self.n_batch, 1], dtype=torch.int64).cuda()            

            if i > 0:                
                start_pt = original_node.gather(1, st_idx.unsqueeze(1).unsqueeze(2).repeat(1,1,4))
                end_pt = original_node.gather(1, idx.unsqueeze(1).unsqueeze(2).repeat(1,1,4)) 
                spt = start_pt[:, :, 2:].squeeze(1)
                ept = end_pt[:, :, :2].squeeze(1) 
                external_reward = torch.norm(ept - spt, dim=1).unsqueeze(1).cuda() # size = [batch,1]
                internal_reward = costs.gather(1, st_idx.unsqueeze(1)) + costs.gather(1, idx.unsqueeze(1))                            
                reward = (external_reward + internal_reward) / 70.0
                # append the reward with zero values when done index is true
                reward = torch.masked_fill(reward, done_index.unsqueeze(1) == 1, 0).squeeze(1)  
                cell_reward.append(reward)
            st_idx = idx.clone()

            _idx = idx.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.n_embedding) # torch.gather를 사용하기 위해 차원을 맞춰줌
            if init_h is None:
                init_h = high_node.gather(1, _idx).squeeze(1)
            h = high_node.gather(1, _idx).squeeze(1)
            h_rest = self.v_weight_embed(torch.cat([init_h, h], dim = -1)) 
            query = h_bar + h_rest               
                
        """
        Ags:
            cell_reward = [batch_size, 1] --> type is tensor
            cell_log_prob = [batch_size, 1] --> type is tensor
            cell_action = [batch_size, num_cells] --> type is list
        """
        cell_log_prob = torch.stack(cell_log_prob, dim = 1)
        cell_reward = torch.stack(cell_reward, dim=1)
        cell_action = torch.stack(cell_action, dim=1)        
        # sum the log_prob and reward
        cell_log_prob = torch.sum(cell_log_prob, dim=-1)        
        cell_reward = torch.sum(cell_reward, dim=-1)

        return cell_log_prob, cell_reward, cell_action                    
    
    def calculate_mean(self, context_vector):
        return torch.mean(context_vector, dim = 1)
# define total model here
class HCPP(torch.nn.Module):
    def __init__(self, 
                n_feature, 
                n_hidden,                 
                n_embedding,                                 
                n_head,
                C):
        super(HCPP, self).__init__()
        self.h_encoder = Encoder(n_feature= n_feature, n_hidden= n_hidden)
        self.h_decoder = Decoder(n_embedding= n_embedding, n_hidden= n_hidden, n_feature=n_feature, n_head=n_head, C = C)        

    def forward(self, map, num_cell, points, costs):     
        high_node, original_node, batch_costs = self.h_encoder(points, costs, mask = None)                          
        high_log_prob, high_reward, high_action = self.h_decoder(high_node, original_node, map, num_cell, batch_costs)
        return high_log_prob, high_reward, high_action


