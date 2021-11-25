import torch
import torch.nn as nn
from Environment import Environment
from module import Pointer, AttentionModule, Glimpse
from torch.distributions import Categorical


class Encoder(torch.nn.Module):
    def __init__(self, n_feature, n_hidden):
        super(Encoder, self).__init__()        
        self.n_hidden = n_hidden
        self.embedding_x = nn.Linear(n_feature, n_hidden)                        
        
        self.high_attention = AttentionModule(
                                            n_heads = 8,
                                            n_hidden = n_hidden,
                                            feed_forward_hidden= 512,
                                            n_layers = 3
                                            )

    def forward(self, batch_data, mask = None):
        self.batch_data = batch_data.cuda()        
    
        # cell embedding for high model             
        self.high_node = self.embedding_x(self.batch_data)        
        self.high_node = self.high_attention(self.high_node)        

        return self.high_node, self.batch_data

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

    def forward(self, cell_context, original_data, high_mask):        
        self.original_data = original_data
        self.batch_size = cell_context.size(0)                
        self.seq_len = self.original_data.size(1)
        self.n_node = self.original_data.size(2)
        init_h = None
        h = None

        # initialize node
        self.init_node = torch.zeros([self.batch_size, 2]).cuda()  
        self.last_node = torch.zeros([self.batch_size, 2]).cuda()  
        
        # calculate query
        h_mean = self.calculate_context(context_vector= cell_context) # batch * 1 * embedding_size
        h_bar = self.h_context_embed(h_mean) # batch * 1 * embedding_size
        h_rest = self.v_weight_embed(self.init_w) # 1 * embedding_size
        query = h_bar + h_rest # kool 논문에서는 concatenate 한다고 나와있지만 실제로는 아님...?

        # set the high environment
        self.high_environment = Environment(batch_data= cell_context, is_local= False)                            
        
        # initialize cell log_prob and cell reward                
        cell_log_prob = []
        cell_reward = []        
        cell_action = []

        for i in range(self.seq_len):                             

            logits = self.high_pointer(query=query, target=cell_context, mask = high_mask)                   
            node_distribtution = Categorical(logits)
            idx = node_distribtution.sample() # idx size = [batch]                                                
            _cell_log_prob = node_distribtution.log_prob(idx)

            # append high action
            cell_action.append(idx)

            # calculate total log probability
            cell_log_prob.append(_cell_log_prob)                                                                                                                               

            # action masking for high level policy
            high_mask = self.high_environment.update_state(action_mask= high_mask, next_idx= idx)            
            
            # get current_node
            self.init_node = self.original_data.gather(1, idx.unsqueeze(1).unsqueeze(2).repeat(1, 1, 2))

            if i > 0:        
                # calculate cell distance    
                _cell_reward = self.calculate_cell_distance(init_node= self.init_node.squeeze(1), last_node= self.last_node.squeeze(1))

                # append cell_reward
                cell_reward.append(_cell_reward)               
            
            self.last_node = self.init_node.clone()

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
        """                
        cell_log_prob = torch.stack(cell_log_prob, 1)
        cell_action = torch.stack(cell_action, 1) 
        cell_reward = torch.stack(cell_reward, 1)                
        # sum up log probability and reward                 
        """
        cell_reward = batch  
        cell_log_prob = batch
        """        
        cell_reward = torch.sum(cell_reward.clone(), dim=-1)
        cell_log_prob = torch.sum(cell_log_prob.clone(), dim=-1)        

        return cell_log_prob, cell_reward, cell_action
                
    def calculate_context(self, context_vector):
        return torch.mean(context_vector, dim=1)

    def calculate_cell_distance(self, init_node, last_node): 
        return torch.norm(last_node - init_node, dim=1).cuda()

class HCPP(torch.nn.Module):
    def __init__(self, 
                n_feature, 
                n_hidden,                 
                n_embedding,                                 
                n_head,
                C):
        super(HCPP, self).__init__()
        self.h_encoder = Encoder(n_feature= n_feature, n_hidden= n_hidden)
        self.h_decoder = Decoder(n_embedding= n_embedding, n_hidden= n_hidden, n_head=n_head, C = C)        

    def forward(self, batch_data, high_mask):             
        cell_embed, original_data = self.h_encoder(batch_data, mask = None)                  
        high_log_prob, high_reward, high_action = self.h_decoder(cell_embed, original_data, high_mask)
        return high_log_prob, high_reward, high_action


