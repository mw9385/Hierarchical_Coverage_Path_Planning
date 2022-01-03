import argparse
import os
import copy
import numpy as np
from numpy.random import shuffle
import torch
import torch.optim as optim
import pprint as pp
import matplotlib.pyplot as plt
import json

from tqdm import tqdm
from Decomposition import Cell
from Mission import CPPDataset, my_collate
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Attention_model import HCPP

# GPU check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device status:{}".format(device))

# main
parser = argparse.ArgumentParser(description="CPP with RL")
parser.add_argument('--epoch', default= 100, help="number of epochs")
parser.add_argument('--steps', default= 2500, help="number of epochs")
parser.add_argument('--batch_size', default=512, help="number of batch size")
parser.add_argument('--n_head', default=8, help="number of heads for multi-head attention")
parser.add_argument('--val_size', default=100, help="number of validation samples") # 이게 굳이 필요한가?
parser.add_argument('--beta', type=float, default=0.9, help="beta")
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
parser.add_argument('--n_nodes', default=9, help='number of visiting nodes')
parser.add_argument('--n_hidden', default=128, help="nuber of hidden nodes") # 
parser.add_argument('--log_interval', default=500, help="store model at every epoch")
parser.add_argument('--eval_interval', default=500, help='update frequency')
parser.add_argument('--log_dir', default='./log/V1', type=str, help='directory for the tensorboard')
parser.add_argument('--file_name', default='HCPP_V1', help='file directory')
parser.add_argument('--test_file_name', default='test_performance/V1/', help='test_file directory')
args = parser.parse_args()
# save args
with open('./model/model_V1.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)
args = vars(args)

learning_rate = args['lr']
steps = args['steps']
B = int(args['batch_size'])
B_val = int(args['val_size'])
n_epoch = int(args['epoch'])
n_nodes = int(args['n_nodes'])
n_hidden = int(args['n_hidden'])
n_head = int(args['n_head'])
log_interval = int(args['log_interval'])
eval_interval = int(args['eval_interval'])
file_name = str(args['file_name'])
test_file_name = str(args['test_file_name'])
beta = float(args['beta'])
# print parameters
pp.pprint(args)

# generate training data
n_train_samples = 10000
n_val_samples = 5000
print("---------------------------------------------")
print("GENERATE DATA")
train_generator = CPPDataset(map_dir='Decomposed data/', pnc_dir='Points and costs/', transform=None)
valid_generator = CPPDataset(map_dir='Decomposed data/', pnc_dir='Points and costs/', transform=None)
train_data_loader = DataLoader(
                    train_generator,
                    batch_size = B,
                    shuffle = True,
                    pin_memory= True, 
                    collate_fn=my_collate)
valid_data_loader = DataLoader(
                    valid_generator,
                    batch_size = B_val,
                    shuffle = True,
                    pin_memory= True,
                    collate_fn=my_collate)                           
print("FINISHED")

# tensorboard 
writer = SummaryWriter(log_dir=args['log_dir'])

# define model
high_model = HCPP(n_feature = 4, n_hidden= n_hidden, n_embedding= n_hidden, n_head=n_head, C = 10).cuda()
optimizer = torch.optim.Adam(high_model.parameters(), lr = learning_rate)

# visualization of tsp results
if __name__=="__main__":
    print("---------------------------------------------")
    print("GENERATE BASELINE")
    for indicies, sample_batch in zip(range(1), train_data_loader):    
        baseline_map = sample_batch[0]
        baseline_num_cells = sample_batch[1]
        baseline_points = sample_batch[2]
        baseline_costs = sample_batch[3]
    
    # get log_prob and reward
    base_high_log_prob, base_high_reward, _ = high_model(baseline_map, baseline_num_cells, baseline_points, baseline_costs)    
    # define initial moving average
    _baseline_high = base_high_reward.clone()    
    print("FINISHED")
    # clear cache
    torch.cuda.empty_cache()
    
    # variable for evaluation
    global_step = 0
    print("---------------------------------------------")
    print("START TRAINING")
    high_model.train()    

    for epoch in range(n_epoch):
        for steps, sample_batch in enumerate(tqdm(train_data_loader)):                
            train_map = sample_batch[0]
            train_num_cells = sample_batch[1]
            train_points = sample_batch[2]
            train_costs = sample_batch[3]
            
            high_log_prob, high_cost, high_action = high_model(train_map, train_num_cells, train_points, train_costs)
            baseline_high = _baseline_high * beta + high_cost * (1.0 - beta)   
            # calculate advantage
            high_advantage = high_cost - baseline_high            
            # update baseline
            _baseline_high = baseline_high.clone()            
            # define loss function                    
            high_loss = (high_advantage * high_log_prob).mean()                                                          

            # set the optimizer zero gradient
            optimizer.zero_grad()   
            high_loss.backward()                                                            
            max_grad_norm = 1.0
            torch.nn.utils.clip_grad_norm_(high_model.parameters(), max_grad_norm, norm_type= 2)            
            optimizer.step()

            total_cost = high_cost.mean().cpu()
            
            # model evaluation
            high_model.eval()            
            if global_step !=0 and global_step % eval_interval == 0:
                print("MODEL EVALUATION")                              
                # generate test valid index
                for indicies, sample_batch in zip(range(1), valid_data_loader):    
                    test_map = sample_batch[0]
                    test_num_cells = sample_batch[1]
                    test_points = sample_batch[2]
                    test_costs = sample_batch[3]

                _, test_high_cost, test_high_action = high_model(test_map, test_num_cells, test_points, test_costs)
                test_total_cost = test_high_cost.mean()
                print("TEST Performance of {}th step:{}".format(global_step, test_total_cost))
                writer.add_scalar("Test Distance", test_total_cost, global_step= global_step)                            
            
            # tensorboard 설정 
            if steps!=0 and steps % log_interval == 0:
                print("SAVE MODEL")
                dir_root = './model/HCPP'
                param_path = dir_root +  "/" + file_name + ".param"
                high_config_path = dir_root + "/" + 'high_' + file_name                

                # make model directory if is not exist
                os.makedirs(dir_root, exist_ok=True)    
                # save high_model            
                torch.save(high_model, high_config_path)                
                
                # write information in tensorboard            
                writer.add_scalar("loss", high_loss, global_step= global_step)
                writer.add_scalar("distance", total_cost, global_step= global_step)

            global_step +=1
            high_model.train()            

    writer.close()   






