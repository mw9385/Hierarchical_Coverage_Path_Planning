import argparse
import copy
import math
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import pprint as pp


from torch.autograd import Variable
from torch.optim import lr_scheduler
from tqdm import tqdm
from scipy.spatial import distance
from generate_data import TSP
from torch.utils.tensorboard import SummaryWriter
from Attention_model import HCPP

# GPU check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device status:{}".format(device))

# main
parser = argparse.ArgumentParser(description="CPP with RL")
parser.add_argument('--size', default=20, help="number of nodes")
parser.add_argument('--epoch', default= 15, help="number of epochs")
parser.add_argument('--batch_size', default= 4, help="number of batch size")
parser.add_argument('--training_step', default= 2500, help="number of training steps")
parser.add_argument('--val_size', default=100, help="number of validation samples") # 이게 굳이 필요한가?
parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
parser.add_argument('--n_cells', default=10, help='number of visiting cells')
parser.add_argument('--max_distance', default=1, help="maximum distance of nodes from the center of cell")
parser.add_argument('--n_hidden', default=128, help="nuber of hidden nodes")
args = vars(parser.parse_args())

size = int(args['size'])
learning_rate = args['lr']
B = int(args['batch_size'])
B_val = int(args['val_size'])
steps = int(args['training_step'])
n_epoch = int(args['epoch'])
n_cells = int(args['n_cells'])
n_hidden = int(args['n_hidden'])
max_distance = int(args['max_distance'])
save_root = './model/cpp_low' + str(size) +'_low.pt'

# print parameters
pp.pprint(args)

# generate training data
n_train_samples = 1000
n_val_samples = 1000
train_tsp_generator = TSP(n_batch=n_train_samples, n_cells = n_cells, size = size, max_distance = max_distance, is_train= True)
valid_tsp_generator = TSP(n_batch=n_val_samples, n_cells = n_cells, size = size, max_distance = max_distance, is_train= False)
[X_train, X_train_cell, X_train_vector] = train_tsp_generator.generate_data()
[X_val, X_val_cell, X_val_vector] = valid_tsp_generator.generate_data()
# tensorboard 
writer = SummaryWriter()

# define state embedding layer
batch_index = np.random.permutation(n_train_samples)
batch_index = batch_index[:B]

# select random batch: list 형태이기 때문에 index vector를 직접적으로 사용하기 어려움
X = [X_train[i] for i in batch_index]

# create a mask
low_mask = [] # low_policy_mask
for sub_x in X:    
    f_mask = []
    for subsub_x in sub_x:        
        num_cities = subsub_x.size(0)
        _mask = torch.zeros((num_cities), dtype = torch.int64).cuda()
        f_mask.append(_mask)
    low_mask.append(f_mask)
# high policy mask
high_mask = torch.zeros([B, n_cells], dtype = torch.int64).cuda()

# state embedding
model = HCPP(n_feature = 2, n_hidden= n_hidden, high_level= True, n_embedding= n_hidden, seq_len= n_cells, C = 10).cuda()
log_prob, reward = model(X, high_mask = high_mask, low_mask = low_mask)






