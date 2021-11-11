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
            action_mask = action_mask.scatter(1, self.next_idx.unsqueeze(1), 1)            
            # scatter 함수는 특정 index의 위치에 원하는 value를 넣을 수 있다. 

            self.next_idx = self.next_idx.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.n_hidden) 
            self.next_node = torch.gather(self.batch_data, dim=1, index = self.next_idx)
                                    
        else: 
            # update the action mask           
            # self.next_idx = self.next_idx.unsqueeze(1)            
            action_mask = action_mask.scatter(1, self.next_idx.unsqueeze(1), 1)    

            # get the next node state              
            self.next_idx = self.next_idx.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.n_hidden)
            self.next_node = torch.gather(self.batch_data, dim=1, index = self.next_idx)            
        return action_mask    

    
