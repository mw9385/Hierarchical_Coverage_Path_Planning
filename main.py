import argparse
import os
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import pprint as pp

from torch.optim import lr_scheduler
from tqdm import tqdm
from generate_data import TSP
from torch.utils.tensorboard import SummaryWriter
from Attention_model import HCPP

# GPU check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device status:{}".format(device))

# main
parser = argparse.ArgumentParser(description="CPP with RL")
parser.add_argument('--size', default=145, help="number of nodes")
parser.add_argument('--epoch', default= 10, help="number of epochs")
parser.add_argument('--steps', default= 100, help="number of epochs")
parser.add_argument('--batch_size', default= 32, help="number of batch size")
parser.add_argument('--val_size', default=100, help="number of validation samples") # 이게 굳이 필요한가?
parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
parser.add_argument('--n_cells', default=10, help='number of visiting cells')
parser.add_argument('--max_distance', default=1, help="maximum distance of nodes from the center of cell")
parser.add_argument('--n_hidden', default=128, help="nuber of hidden nodes")
parser.add_argument('--log_interval', default= 1, help="store model at every epoch")
args = vars(parser.parse_args())

size = int(args['size'])
learning_rate = args['lr']
steps = args['steps']
B = int(args['batch_size'])
B_val = int(args['val_size'])
n_epoch = int(args['epoch'])
n_cells = int(args['n_cells'])
n_hidden = int(args['n_hidden'])
max_distance = int(args['max_distance'])
log_interval = int(args['log_interval'])

# print parameters
pp.pprint(args)

# generate training data
n_train_samples = 50000
n_val_samples = 10000
print("---------------------------------------------")
print("GENERATE DATA")
train_tsp_generator = TSP(n_batch=n_train_samples, n_cells = n_cells, size = size, max_distance = max_distance, is_train= True)
valid_tsp_generator = TSP(n_batch=n_val_samples, n_cells = n_cells, size = size, max_distance = max_distance, is_train= False)
[X_train, X_train_cell, X_train_vector] = train_tsp_generator.generate_data()
[X_val, X_val_cell, X_val_vector] = valid_tsp_generator.generate_data()
print("FINISHED")

# tensorboard 
writer = SummaryWriter(log_dir='./log')

# define model
model = HCPP(n_feature = 2, n_hidden= n_hidden, high_level= True, n_embedding= n_hidden, seq_len= n_cells, C = 10).cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
lr_decay_step = 2500
lr_decay_rate = 0.96
opt_scheduler = lr_scheduler.MultiStepLR(optimizer, range(lr_decay_step, lr_decay_step*1000,
                                                          lr_decay_step), gamma=lr_decay_rate)
beta = 0.8

if __name__=="__main__":
    print("---------------------------------------------")
    print("GENERATE BASELINE")
    
    # generate samples for baseline
    batch_index = np.random.permutation(n_train_samples)
    batch_index = batch_index[:B]
    baseline_X = [X_train[i] for i in batch_index]

    # generate mask for baseline
    low_mask = [] # low_policy_mask
    for sub_x in baseline_X:    
        f_mask = []
        for subsub_x in sub_x:        
            num_cities = subsub_x.size(0)
            _mask = torch.zeros((num_cities), dtype = torch.int64).cuda()
            f_mask.append(_mask)
        low_mask.append(f_mask)
    # high policy mask
    high_mask = torch.zeros([B, n_cells], dtype = torch.int64).cuda()

    # get log_prob and reward
    base_log_prob, base_reward = model(baseline_X, high_mask = high_mask, low_mask = low_mask)

    # define initial moving average
    moving_avg = base_reward.clone()    
    print("FINISHED")
    # clear cache
    torch.cuda.empty_cache()

    print("---------------------------------------------")
    print("START TRAINING")
    model.train()
    for epoch in range(n_epoch):        
        for step in tqdm(range(steps)):
            # define state embedding layer
            batch_index = np.random.permutation(n_train_samples)
            batch_index = batch_index[:B]
            # train data            
            X = [X_train[i] for i in batch_index]

            # -----------------------------------------------------#
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
            # -----------------------------------------------------#

            log_prob, reward = model(X, high_mask = high_mask, low_mask = low_mask)  

            moving_avg = moving_avg * beta + reward * (1.0 - beta)
            advantage =  reward -  moving_avg
                        
            loss = (advantage * log_prob).mean()
            loss.backward()
            optimizer.zero_grad()

            max_grad_norm = 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, norm_type= 2)
            optimizer.step()
            opt_scheduler.step()

        # tensorboard 설정 
        if epoch % log_interval == 0:
            print("SAVE MODEL")
            dir_root = './model/HCPP'
            file_name = "HCPP_V1"
            param_path = dir_root +  "/" + file_name + ".param"
            config_path = dir_root + "/" + file_name + '.config'

            # make model directory if is not exist
            os.makedirs(dir_root, exist_ok=True)
                        
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'reward': reward
            }, config_path)

            # write information in tensorboard            
            writer.add_scalar("loss", loss, global_step=epoch)
            writer.add_scalar("distance", reward, global_step= epoch)
                        
                # model evaluation 설정
    writer.close()   






