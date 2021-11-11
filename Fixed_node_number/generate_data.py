import numpy as np
import torch
import torch.distributions as tdist
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
class TSP():
    def __init__(self, n_batch, n_cells, size, max_distance, is_train):
        self.n_batch = n_batch
        self.n_cells = n_cells
        self.size = size
        self.max_distance = max_distance
        self.is_train = is_train
        self.cell_distance = 100

        if is_train:
            torch.manual_seed(1234)
        else:
            torch.manual_seed(1234)    
        
    def generate_data(self):        
        # generate random cell points
        # fix the number of points 
        self.cell = torch.normal(10, self.cell_distance, size = (self.n_batch, self.n_cells, 2))
        # self.cell = torch.randn(self.n_batch, self.n_cells, 2) * self.cell_distance
        self.tsp_data = []

        for j in range(self.n_batch):
            if j % 1000 == 0:
                print(j)
            _tsp_data = []
            n_rand_cell = self.n_cells
            for i in range(self.n_cells):
                # create a multivariable distributions
                m = tdist.multivariate_normal.MultivariateNormal(self.cell[j, i].clone(), torch.eye(2) * self.max_distance)                
                _sample = m.sample((n_rand_cell,))
                _tsp_data.append(_sample / 50.0)                
            _tsp_data = torch.stack(_tsp_data, 1)
            self.tsp_data.append(_tsp_data)                        
        self.tsp_data = torch.stack(self.tsp_data, 1)     
        self.tsp_data = torch.permute(self.tsp_data, (1, 2, 0, 3))

        return self.tsp_data
    
    def __len__(self):
        return len(self.tsp_data)

    def __getitem__(self, idx):
        return [self.tsp_data[idx], self.cell[idx]]

    def draw(self, idx):            
        sample = self.tsp_data[idx]
        cell_sample = self.cell[idx]
        
        # flatten a list with different length (3D shape) to a 2D vector
        flatten_sample = torch.tensor(())
        for sublist in sample:            
            flatten_sample = torch.cat((flatten_sample, sublist), dim= 0)            
            
        print(flatten_sample.size())
        cell_sample = cell_sample.view(self.n_cells, 2)        
        
        plt.figure()
        plt.scatter(flatten_sample[:,0], flatten_sample[:,1], color ='r', label = 'nodes')
        plt.scatter(cell_sample[:,0], cell_sample[:,1], color = 'b', label = 'center')
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == "__main__":
    tsp = TSP(n_batch=2, n_cells=10, size = 1000, max_distance=10.0, is_train= True)
    [data, cell_data] = tsp.generate_data()
    # print(tsp.__len__())
    # print(tsp.__getitem__(5))
    # draw the sample
    tsp.draw(0)