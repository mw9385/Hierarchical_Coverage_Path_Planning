import numpy as np
import torch
import torch.distributions as tdist
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
class TSP():
    def __init__(self, n_batch, n_nodes, is_train):
        self.n_batch = n_batch              
        self.n_nodes = n_nodes        
        self.is_train = is_train        

        if is_train:
            torch.manual_seed(1234)
        else:
            torch.manual_seed(4321)    
        
    def generate_data(self):                
        _tsp_data = []
        n_node = self.n_nodes
        for i in range(self.n_batch):
            if i%1000 == 0:
                print(i)
            # create a multivariable distributions
            m = tdist.multivariate_normal.MultivariateNormal(torch.zeros(2), torch.eye(2))                
            _sample = m.sample((n_node, ))
            _tsp_data.append(_sample)                                                
        self.tsp_data = torch.stack(_tsp_data, 0)  
        self.tsp_data[:, 0, :] = 0.0
        return self.tsp_data

if __name__ == "__main__":
    tsp = TSP(n_batch=100, n_nodes=10, is_train= True) 
    tsp.generate_data()   