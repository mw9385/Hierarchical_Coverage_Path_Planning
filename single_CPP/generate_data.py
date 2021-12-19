import numpy as np
import torch
import torch.distributions as tdist
import matplotlib.pyplot as plt

class TSP():
    def __init__(self, n_batch, n_nodes, is_train):
        self.n_batch = n_batch              
        self.n_nodes = n_nodes        
        self.is_train = is_train        

        if is_train:
            torch.manual_seed(1234)
        else:
            torch.manual_seed(4321)    
    # generate tsp training data with in range of [20m * 20m]
    def generate_data(self):                
        sample = np.zeros([self.n_batch, self.n_nodes, 2])
        for i in range(self.n_batch):                                    
            if i%1000 == 0:
                print(i)
            for j in range(self.n_nodes):                                
                sample[i, j, 0] = np.random.randint(20) /100.0
                sample[i, j, 1] = np.random.randint(20) /100.0                                                               
        sample = torch.from_numpy(sample).float()        
        return sample

if __name__ == "__main__":
    tsp = TSP(n_batch=100, n_nodes=10, is_train= True) 
    tsp.generate_data()   
