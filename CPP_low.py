import argparse
import copy
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import pprint as pp


from torch.autograd import Variable
from torch.optim import lr_scheduler
from tqdm import tqdm
from gpn import GPN, Embedding, MultiHeadAttentionLayer
from scipy.spatial import distance
from generate_data import TSP
from torch.utils.tensorboard import SummaryWriter

# GPU check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device status:{}".format(device))

# main
parser = argparse.ArgumentParser(description="CPP with RL")
parser.add_argument('--size', default=20, help="number of nodes")
parser.add_argument('--epoch', default= 15, help="number of epochs")
parser.add_argument('--batch_size', default= 32, help="number of batch size")
parser.add_argument('--training_step', default= 2500, help="number of training steps")
parser.add_argument('--val_size', default=100, help="number of validation samples") # 이게 굳이 필요한가?
parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
parser.add_argument('--n_cells', default=10, help='number of visiting cells')
parser.add_argument('--max_distance', default=1, help="maximum distance of nodes from the center of cell")
args = vars(parser.parse_args())

size = int(args['size'])
learning_rate = args['lr']
B = int(args['batch_size'])
B_val = int(args['val_size'])
steps = int(args['training_step'])
n_epoch = int(args['epoch'])
n_cells = int(args['n_cells'])
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

# load model
model = GPN(n_feature=2, n_hidden=128).cuda()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
# define state embedding layer
embedding_all = Embedding(2, 128, init=False)
embedding_init = Embedding(2, 128, init=True)

multi_head_attention = MultiHeadAttentionLayer(
                                            d_model= 128,
                                            n_head = 8,
                                            qkv_fc_layer= nn.Linear(128, 128),
                                            fc_layer = nn.Linear(128, 128))

lr_decay_step = 2500
lr_decay_rate = 0.96
opt_scheduler = lr_scheduler.MultiStepLR(optimizer, range(lr_decay_step, lr_decay_step * 1000,
                                                            lr_decay_step), gamma=lr_decay_rate)

# main loop
for epoch in range(n_epoch):
    for step in tqdm(range(steps)):
        # batch initialization
        batch_index = np.random.permutation(n_train_samples)
        batch_index = batch_index[:B]

        # initialize optmizer
        optimizer.zero_grad()

        # select random batch: list 형태이기 때문에 index vector를 직접적으로 사용하기 어려움
        X = [X_train[i] for i in batch_index]
        X_vector = [X_train_vector[i] for i in batch_index]

        # initial position
        x = torch.tensor(()).cuda()
        for sub_x in X_vector:
            x = torch.cat((x, sub_x[0].unsqueeze(0).cuda()), dim = 0) # size = [batch, 2]
        
        # state embedding
        embedding_X = embedding_all(X) # size = [batch, n_nodes, n_hidden]
        embedding_x = embedding_init(x) # size = [batch, n_hidden]
    
        # number of size
        size = embedding_X.size(1)
        # define mask for action masking
        mask = torch.zeros(B, size).cuda()
        # initialize parameters
        R = 0; logprobs = 0; reward = 0

        # define reference matrix to calculate reward
        h = None; c = None
        for k in range(size):            
            output, h, c, _ = model(x=embedding_x, X_all=embedding_X, h=h, c=c, mask=mask)
            sampler = torch.distributions.Categorical(output)
            # sampler.sample(): 확률 분포에서 확률이 높은 지점 근처에서 sampling
            idx = sampler.sample() # idx size = [batch_size]
            
            Y1 = torch.tensor(())
            for i, (index, nodes) in enumerate(zip(idx, X_vector)):
                Y1 = torch.cat((Y1, nodes[index].unsqueeze(0)), dim = 0)

            if k == 0:
                Y_init = Y1.clone() # receive 0 reward
            else:
                # torch.norm(): return the matrix norm or vetor norm
                reward = torch.norm(Y1-Y0, dim=1).cuda()
            
            Y0 = Y1.clone()
            # agent가 node를 이동함
            x = torch.tensor(()).cuda()
            for i, (index, nodes) in enumerate(zip(idx, X_vector)):
                x = torch.cat((x, nodes[index].unsqueeze(0).cuda()), dim = 0)
            embedding_x = embedding_init(x)

            R += reward
                
            TINY = 1e-15
                                   
            logprobs += torch.log(output[[i for i in range(B)], idx.data]+TINY) 
            # action masking: 이미 방문한 지점들에 대해서는 -inf를 넣음
            mask[[i for i in range(B)], idx.data] += -np.inf                 

        R += torch.norm(Y1-Y_init, dim=1).cuda()

        # self-critic baseline
        mask = torch.zeros(B, size).cuda()

        C = 0; baseline = 0; h = None; c = None;
        
        # initial position
        x = torch.tensor(()).cuda()
        for sub_x in X_vector:
            x = torch.cat((x, sub_x[0].unsqueeze(0).cuda()), dim = 0) # size = [batch, 2]
        # state embedding
        embedding_x = embedding_init(x)

        for k in range(size):
            output, h, c, _ = model(x = embedding_x, X_all = embedding_X, h=h, c=c, mask=mask)
            # get the maximum value: greedy baseline
            idx = torch.argmax(output, dim=1)

            # get the next node position 
            Y1 = torch.tensor(()).cuda()
            for i, (index, nodes) in enumerate(zip(idx, X_vector)):
                Y1 = torch.cat((Y1, nodes[index].unsqueeze(0).cuda()), dim = 0)

            if k == 0:
                Y_init = Y1.clone()
            else:
                baseline = torch.norm(Y1-Y0, dim = 1)

            Y0 = Y1.clone()

            x = torch.tensor(()).cuda()
            for i, (index, nodes) in enumerate(zip(idx, X_vector)):
                x = torch.cat((x, nodes[index].unsqueeze(0).cuda()), dim = 0)
            embedding_x = embedding_init(x)            

            C +=baseline
            mask[[i for i in range(B)], idx.data] += -np.inf
        
        C += torch.norm(Y1-Y_init, dim = 1).cuda()
        gap = (R-C).mean()

        # REINFORCE loss function with baseline
        loss = ((R-C-gap)*logprobs).mean()
        loss.backward()

        max_grad_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, norm_type=2)
        optimizer.step()
        opt_scheduler.step()

        # periodic evaluation
        if step % 50 == 0:
            print("epoch:{}, batch:{}/{}, reward:{}".format(epoch, step, steps, R.mean()))
            
            # greedy evaluation
            tour_len = 0
            
            # get the validation dataset
            val_batch_index = np.random.permutation(n_val_samples)
            val_batch_index = val_batch_index[:B_val]

            x_val = [X_val[i] for i in val_batch_index]
            x_val_vector = [X_val_vector[i] for i in val_batch_index]
            
            mask = torch.zeros(B_val, size).cuda()
            # initialize values
            R = 0; logprobs = 0; reward = 0;
            
            # initial position
            x_val_init = torch.tensor(()).cuda()
            for sub_x in x_val_vector:
                x_val_init = torch.cat((x_val_init, sub_x[0].unsqueeze(0).cuda()), dim = 0) # size = [batch, 2]
            
            # state embedding
            embedding_x_val = embedding_all(x_val) # size = [batch, n_nodes, n_hidden]
            embedding_x = embedding_init(x_val_init) # size = [batch, n_hidden]

            h = None; c = None;

            for k in range(size):                
                output, h, c, _ = model(x = embedding_x, X_all = embedding_x_val, h=h, c=c, mask = mask)
                idx = torch.argmax(output, dim=1)
                
                # get the next node position 
                Y1 = torch.tensor(()).cuda()
                for i, (index, nodes) in enumerate(zip(idx, x_val_vector)):
                    Y1 = torch.cat((Y1, nodes[index].unsqueeze(0).cuda()), dim = 0)
                
                if k == 0:
                    Y_init = Y1.clone()
                else:
                    reward = torch.norm(Y1-Y0, dim = 1).cuda()
                Y0 = Y1.clone()
                
                x = torch.tensor(()).cuda()
                for i, (index, nodes) in enumerate(zip(idx, x_val_vector)):
                    x = torch.cat((x, nodes[index].unsqueeze(0).cuda()), dim = 0)            
                
                R += reward
                mask[[i for i in range(B_val)], idx.data] += -np.inf

            R += torch.norm(Y1-Y_init, dim = 1)
            tour_len +=R.mean().item()
            print('Validation tour length:', tour_len)
                
    print('save model to: ', save_root)
    torch.save(model, save_root)
