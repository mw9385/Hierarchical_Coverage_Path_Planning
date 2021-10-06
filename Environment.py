import torch

class Environment():
    def __init__(self, batch_data, is_local):
        self.batch_data = batch_data
        self.is_local = is_local
        self.n_hidden = self.batch_data.size(2)
        self.n_batch = self.batch_data.size(0)

    def update_state(self, action_mask, next_idx):        
        self.action_mask = action_mask                
        self.next_idx = next_idx

        assert action_mask is not None, "Action mask is Empty" 
        assert self.batch_data is not None, "Batch data is Empty"

        # update states for local nodes
        if self.is_local is True:                        
            action_mask = action_mask.scatter(0, self.next_idx.data, 1)            
            # action_mask[self.next_idx.data] =1 # 이렇게 하면 잘 안됌. 이유가 뭘까

            self.next_idx = self.next_idx.unsqueeze(1).repeat(1, self.n_hidden)
            for cells in self.batch_data:
                for local_nodes in cells:                    
                    local_nodes = local_nodes.unsqueeze(0)                    
                    self.next_node = torch.gather(local_nodes, dim=1, index=self.next_idx)                    
                                    
        else: 
            # update the action mask           
            # self.next_idx = self.next_idx.unsqueeze(1)            
            action_mask = action_mask.scatter(1, self.next_idx.unsqueeze(1), 1)    

            # get the next node state              
            self.next_idx = self.next_idx.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.n_hidden)
            self.next_node = torch.gather(self.batch_data, dim=1, index = self.next_idx)            
        return action_mask
    
    # def calculate_distance(self):
