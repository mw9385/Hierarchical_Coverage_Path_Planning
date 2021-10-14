import argparse
import os
import numpy as np
import torch
# import matplotlib.pyplot as plt
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
parser.add_argument('--steps', default= 500, help="number of epochs")
parser.add_argument('--batch_size', default=512, help="number of batch size")
parser.add_argument('--val_size', default=100, help="number of validation samples") # 이게 굳이 필요한가?
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
parser.add_argument('--n_cells', default=5, help='number of visiting cells')
parser.add_argument('--max_distance', default=20, help="maximum distance of nodes from the center of cell")
parser.add_argument('--n_hidden', default=128, help="nuber of hidden nodes") # 512개를 사용하는 경우 성능이 좋지 못함
parser.add_argument('--log_interval', default=5, help="store model at every epoch")
parser.add_argument('--eval_interval', default=50, help='update frequency')
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
eval_interval = int(args['eval_interval'])

# print parameters
pp.pprint(args)

# generate training data
n_train_samples = 50000
n_val_samples = 1000
print("---------------------------------------------")
print("GENERATE DATA")
train_tsp_generator = TSP(n_batch=n_train_samples, n_cells = n_cells, size = size, max_distance = max_distance, is_train= True)
valid_tsp_generator = TSP(n_batch=n_val_samples, n_cells = n_cells, size = size, max_distance = max_distance, is_train= False)
X_train = train_tsp_generator.generate_data()
X_val = valid_tsp_generator.generate_data()
print("FINISHED")

# tensorboard 
writer = SummaryWriter(log_dir='./log/V3')

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
    base_log_prob, base_reward, _, _ = model(baseline_X, high_mask = high_mask, low_mask = low_mask)    

    # define initial moving average
    _baseline = base_reward.clone()    
    print("FINISHED")
    # clear cache
    torch.cuda.empty_cache()
    
    # variable for evaluation
    global_step = 0
    print("---------------------------------------------")
    print("START TRAINING")
    model.train()
    for epoch in range(n_epoch):        
        for step in tqdm(range(steps)):    
            # set the optimizer zero gradient
            optimizer.zero_grad()
            # define state embedding layer
            batch_index = np.random.permutation(n_train_samples)
            batch_index = batch_index[:B]
            # train data            
            X = [X_train[i] for i in batch_index]

            # ----------------------------------------------------------------------------------------------------------#
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
            # ----------------------------------------------------------------------------------------------------------#

            log_prob, cost, _, _ = model(X, high_mask = high_mask, low_mask = low_mask)                  
            baseline = _baseline * beta + cost * (1.0 - beta)
            advantage = cost - baseline
            _baseline = baseline.clone()
        
            loss = (advantage * log_prob).mean()                                                
            loss.backward()                                
                                    
            max_grad_norm = 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, norm_type= 2)
            optimizer.step()
            opt_scheduler.step()
            
            # model evaluation
            model.eval()
            if global_step !=0 and global_step % eval_interval == 0:
                print("MODEL EVALUATION")                              
                # generate test valid index
                test_batch_index = np.random.permutation(n_val_samples)
                test_batch_index = test_batch_index[:B]
                
                # get test data from test batch
                test_X = [X_val[i] for i in test_batch_index]

                # ----------------------------------------------------------------------------------------------------------#
                # create a mask for test
                test_low_mask = [] # low_policy_mask
                for sub_x in test_X:    
                    f_mask = []
                    for subsub_x in sub_x:        
                        num_cities = subsub_x.size(0)
                        _mask = torch.zeros((num_cities), dtype = torch.int64).cuda()
                        f_mask.append(_mask)
                    test_low_mask.append(f_mask)
                # high policy mask
                test_high_mask = torch.zeros([B, n_cells], dtype = torch.int64).cuda()
                # ----------------------------------------------------------------------------------------------------------#
                # evaluate the performance 
                _, test_cost, _, _ = model(test_X, high_mask = test_high_mask, low_mask = test_low_mask)  
                test_cost = test_cost.mean()
                print("TEST Performance of {}th step:{}".format(global_step, test_cost))
                writer.add_scalar("Test Distance", test_cost, global_step= global_step)                            

            # tensorboard 설정 
            if step!=0 and step % log_interval == 0:
                print("SAVE MODEL")
                dir_root = './model/HCPP'
                file_name = "HCPP_V3"
                param_path = dir_root +  "/" + file_name + ".param"
                config_path = dir_root + "/" + file_name + '.config'

                # make model directory if is not exist
                os.makedirs(dir_root, exist_ok=True)
                # torch.save(model, model.state_dict(), config_path)  
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'cost': cost.mean()
                }, config_path)

                # write information in tensorboard            
                writer.add_scalar("loss", loss, global_step= global_step)
                writer.add_scalar("distance", cost.mean(), global_step= global_step)

                # write gradient information
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.grad, global_step= global_step)
            global_step +=1
            model.train()

    writer.close()   






