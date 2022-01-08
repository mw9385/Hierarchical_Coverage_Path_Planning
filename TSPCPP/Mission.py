import numpy as np
import random
import os
import pickle
import torch
from torch._C import dtype

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
def my_collate(batch):
    _map, _num_cells, new_points, new_costs, new_paths = [], [], [], [], []
    for index, samples in enumerate(batch):
        map = samples['map']
        num_cells = samples['num_cells']
        points = samples['points']
        costs = samples['costs']
        paths = samples['paths']

        _map.append(map)
        _num_cells.append(num_cells)
        new_points.append(points)
        new_costs.append(costs)
        new_paths.append(paths)
        
    new_map = torch.stack(_map, dim=0)
    new_num_cells = torch.stack(_num_cells, dim=0)    
    return [new_map, new_num_cells, new_points, new_costs, new_paths]

class Cell :
    def __init__(self) :
        self.min_x, self.max_x = None, None
        self.left, self.right = list(), list()
        self.ceiling, self.floor = dict(), dict()
        self.center = (None, None)

class CPPDataset(Dataset):
    """Load stored decomposed maps, start points, end points, and costs"""
    def __init__(self, map_dir, pnc_dir, transform = None):
        """
        Args:
            map_dir: directory for stored decomposed maps
            pnc_dir: directory for points and costs (pnc) directory
        """
        self.map_dir = map_dir
        self.pnc_dir = pnc_dir
        self.transform = transform
    
    def __len__(self):
        list = os.listdir(self.map_dir)
        num_files = len(list)
        return num_files
    
    def __getitem__(self, idx):
        # all elements 1) should be tensor and 2) should have same length.
        if torch.is_tensor(idx):
            idx = idx.tolist()
        map, num_cells, _ = pickle.load(open(self.map_dir + "decomposed_" + str(idx), 'rb'))
        points, costs, paths = pickle.load(open(self.pnc_dir + "PNC_" + str(idx), 'rb'))    
        points = torch.from_numpy(points) 
        costs = torch.from_numpy(costs)
        paths = torch.from_numpy(paths)        
        decomposed_image = torch.from_numpy(map)

        # resize the shape
        points = points.unsqueeze(0)
        costs = costs.unsqueeze(0)
        paths = paths.unsqueeze(0)
        num_cells = torch.tensor([num_cells]) 

        # convert the decomposed image into a binary image
        A, B = decomposed_image.size()
        decomposed_image = np.ones([A, B, 3], dtype = np.uint8)
        decomposed_image[map > 0, :] = [255, 255, 255]
        decomposed_image[decomposed_image > 127] = 0
        if len(decomposed_image.shape) > 2: 
            decomposed_image = decomposed_image[:, :, 0]        
        decomposed_image = torch.from_numpy(decomposed_image)

        # return the dataset
        sample = {'map': decomposed_image, 'num_cells': num_cells, 'points': points, 'costs': costs, 'paths': paths}        
        return sample

if __name__ == "__main__":
    Dataset = CPPDataset(map_dir= "./Decomposed_data/", pnc_dir= "./Points_and_costs/", transform=None)
    for index, sample in enumerate(Dataset):
        print(index)