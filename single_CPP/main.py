import argparse
import os
import copy
import numpy as np
import torch
import torch.optim as optim
import pprint as pp
import matplotlib.pyplot as plt
import json

from tqdm import tqdm
from generate_data import TSP
from torch.utils.tensorboard import SummaryWriter
from Attention_model import HCPP

# GPU check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device status:{}".format(device))

# main
parser = argparse.ArgumentParser(description="CPP with RL")
parser.add_argument('--epoch', default= 100, help="number of epochs")
parser.add_argument('--steps', default= 2500, help="number of epochs")
parser.add_argument('--batch_size', default=2048, help="number of batch size")
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
with open('./model/HCPP/model_V1.txt', 'w') as f:
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
train_tsp_generator = TSP(n_batch=n_train_samples, n_nodes = n_nodes, is_train= True)
valid_tsp_generator = TSP(n_batch=n_val_samples, n_nodes = n_nodes, is_train= False)
X_train = train_tsp_generator.generate_data()
X_val = valid_tsp_generator.generate_data()
print("FINISHED")

# tensorboard 
writer = SummaryWriter(log_dir=args['log_dir'])

# define model
high_model = HCPP(n_feature = 2, n_hidden= n_hidden, n_embedding= n_hidden, n_head=n_head, C = 10).cuda()
optimizer = torch.optim.Adam(high_model.parameters(), lr = learning_rate)

# visualization of tsp results
def plot_tsp(sample, high_mask, test_file_name, global_step):        
    high_mask = high_mask.cpu().numpy()
    plt.figure()        
    plt.grid()

    R = 0
    for i, high_index in enumerate(high_mask):    
        points = sample[high_index]                
         
        if i == 0:
            plt.text(points[0], points[1], 'depot', fontsize=10, label='depot')
        else:
            plt.scatter(points[0], points[1], s=10)
            plt.plot([_points[0], points[0]], [_points[1], points[1]], color='black')
            plt.text(points[0], points[1], i, fontsize = 10)
            _reward = calculate_distance(_points, points)
            R +=_reward
        _points = copy.deepcopy(points)
        plt.pause(0.01)
    plt.title('Total distance:{}'.format(R))
    plt.show(block = False)
    plt.savefig(test_file_name + 'route_' + str(global_step), bbox_inches = 'tight', dpi='figure')
    plt.close('all')
# calculate distance between two nodes    
def calculate_distance(input_node, output_node):
    return torch.norm(input_node - output_node, dim=0)

if __name__=="__main__":
    print("---------------------------------------------")
    print("GENERATE BASELINE")
                    
    # generate samples for baseline
    batch_index = np.random.permutation(n_train_samples)
    batch_index = batch_index[:B]
    baseline_X = X_train[batch_index]    

    # generate mask for baseline           
    high_mask = torch.zeros([B, n_nodes], dtype = torch.int64).cuda()

    # get log_prob and reward
    base_high_log_prob, base_high_reward, _ = high_model(baseline_X, high_mask = high_mask)    
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
        for step in tqdm(range(steps)):    
         
            # define state embedding layer
            batch_index = np.random.permutation(n_train_samples)
            batch_index = batch_index[:B]
            X = X_train[batch_index]    

            # ----------------------------------------------------------------------------------------------------------#            
            # define action masks for low and high model            
            high_mask = torch.zeros([B, n_nodes], dtype = torch.int64).cuda()
            # ----------------------------------------------------------------------------------------------------------#
            high_log_prob, high_cost, high_action = high_model(X, high_mask = high_mask)            
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
                test_batch_index = np.random.permutation(n_val_samples)
                test_batch_index = test_batch_index[:B]
                
                # get test data from test batch
                test_X = X_val[test_batch_index]

                # ----------------------------------------------------------------------------------------------------------#
                # create a mask for test                                                
                test_high_mask = torch.zeros([B, n_nodes], dtype = torch.int64).cuda()
                # ----------------------------------------------------------------------------------------------------------#
                # evaluate the performance                 
                _, test_high_cost, test_high_action = high_model(test_X, high_mask = test_high_mask)                                  
                test_total_cost = test_high_cost.mean()
                print("TEST Performance of {}th step:{}".format(global_step, test_total_cost))
                writer.add_scalar("Test Distance", test_total_cost, global_step= global_step)                            

                # visualization                 
                random_batch_index = np.random.randint(B)
                sample = test_X[random_batch_index]
                sample_high_action = test_high_action[random_batch_index]                
                
                plot_tsp(sample=sample, high_mask=sample_high_action, test_file_name=test_file_name, global_step = global_step)



            # tensorboard 설정 
            if step!=0 and step % log_interval == 0:
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






