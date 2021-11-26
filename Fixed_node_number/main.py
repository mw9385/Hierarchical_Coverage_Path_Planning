import argparse
import os
import copy
import numpy as np
import torch
# import matplotlib.pyplot as plt
import torch.optim as optim
import pprint as pp
import matplotlib.pyplot as plt
import json

from tqdm import tqdm
from generate_data import TSP
from torch.utils.tensorboard import SummaryWriter
from Attention_model import HCPP, Low_Decoder

# GPU check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device status:{}".format(device))

# main
parser = argparse.ArgumentParser(description="CPP with RL")
# training steps
parser.add_argument('--epoch', default= 100, help="number of epochs")
parser.add_argument('--steps', default= 2500, help="number of epochs")
# hyperparameters
parser.add_argument('--batch_size', default=512, help="number of batch size")
parser.add_argument('--val_size', default=100, help="number of validation samples") # 이게 굳이 필요한가?
parser.add_argument('--beta', type=float, default=0.9, help="beta")
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
parser.add_argument('--n_layer', default=3, help='number of visiting nodes')
parser.add_argument('--n_hidden', default=128, help="nuber of hidden nodes") # 
# data generation
parser.add_argument('--n_head', default=8, help="number of heads for multi-head attention")
parser.add_argument('--n_cells', default=5, help='number of visiting cells')
parser.add_argument('--n_nodes', default=10, help='number of visiting nodes')
parser.add_argument('--max_distance', default=1000, help="maximum distance of nodes from the center of cell")
parser.add_argument('--node_distance', default=1000, help='maximum distance between each cells')
# evaluation
parser.add_argument('--log_interval', default=500, help="store model at every epoch")
parser.add_argument('--eval_interval', default=500, help='update frequency')
# paths
parser.add_argument('--log_dir', default='./log/V2', type=str, help='directory for the tensorboard')
parser.add_argument('--file_name', default='HCPP_V2', help='file directory')
parser.add_argument('--test_file_name', default='test_performance/V2/', help='test_file directory')
args = parser.parse_args()
# save args
with open('./model/HCPP/model_V2.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)
args = vars(args)

learning_rate = args['lr']
steps = args['steps']
B = int(args['batch_size'])
B_val = int(args['val_size'])
n_epoch = int(args['epoch'])
n_cells = int(args['n_cells'])
n_nodes = int(args['n_nodes'])
n_hidden = int(args['n_hidden'])
n_head = int(args['n_head'])
n_layer =int(args['n_layer'])
max_distance = int(args['max_distance'])
node_distance = int(args['node_distance'])
log_interval = int(args['log_interval'])
eval_interval = int(args['eval_interval'])
file_name = str(args['file_name'])
test_file_name = str(args['test_file_name'])
beta = float(args['beta'])

# print parameters
pp.pprint(args)

# generate training data
n_train_samples = 100000
n_val_samples = 1000
print("---------------------------------------------")
print("GENERATE DATA")
train_tsp_generator = TSP(n_batch=n_train_samples, n_cells = n_cells, n_nodes = n_nodes, max_distance = max_distance, node_distance = node_distance, is_train= True)
valid_tsp_generator = TSP(n_batch=n_val_samples, n_cells = n_cells, n_nodes = n_nodes, max_distance = max_distance, node_distance = node_distance, is_train= False)
X_train = train_tsp_generator.generate_data()
X_val = valid_tsp_generator.generate_data()
print("FINISHED")

# tensorboard 
writer = SummaryWriter(log_dir=args['log_dir'])

# define model
low_model = Low_Decoder(n_embedding= n_hidden, n_hidden=n_hidden, n_head = n_head, C = 10).cuda()
high_model = HCPP(n_feature = 2, n_hidden= n_hidden, n_embedding= n_hidden, n_head=n_head, n_layer=n_layer, C = 10).cuda()
all_params = list(low_model.parameters()) + list(high_model.parameters())
optimizer = torch.optim.Adam(all_params, lr = learning_rate)

# visualization code
def calculate_distance(input_node, output_node):
    return torch.norm(input_node - output_node, dim=0)

def plot_tsp(sample, high_mask, low_mask, test_file_name, global_step):    
    
    plt.figure()            
    high_mask = high_mask.cpu().numpy()
    plt.grid()

    R = 0
    for i, high_index in enumerate(high_mask):                        
        current_cell = sample[high_index]          
        low_index = low_mask[i]
        low_index = low_index.cpu().numpy()     
        plt.scatter(current_cell[:,0], current_cell[:,1], s=30)
        for j in range(len(current_cell)):
            _low_index = np.where(low_index == j)[0]
            points = current_cell[_low_index][0]   

            if i==0 and j==0:                
                # set the depot points
                plt.text(points[0], points[1], 'depot', fontsize=10, label='depot') 
            else:
                plt.plot([_points[0], points[0]], [_points[1], points[1]], color='black')
                plt.text(points[0], points[1], j, fontsize=10)
                _reward = calculate_distance(_points, points)
                R += _reward
            _points = copy.deepcopy(points)
            plt.pause(0.01)        
    plt.title('Total distance:{}'.format(R))
    plt.show(block=False)
    plt.savefig(test_file_name + 'route_' + str(global_step), bbox_inches = 'tight', dpi=1200)
    plt.close('all')

if __name__=="__main__":
    print("---------------------------------------------")
    print("GENERATE BASELINE")
                    
    # generate samples for baseline
    batch_index = np.random.permutation(n_train_samples)
    batch_index = batch_index[:B]
    baseline_X = X_train[batch_index]    

    # generate mask for baseline        
    low_mask = torch.zeros([B, n_cells, n_nodes], dtype= torch.int64).cuda()
    high_mask = torch.zeros([B, n_cells], dtype = torch.int64).cuda()

    # get log_prob and reward
    base_high_log_prob, base_low_log_prob, base_high_reward, base_low_reward, _, _ = high_model(baseline_X, high_mask = high_mask, low_mask = low_mask, low_decoder = low_model)    
    
    # define initial moving average
    _baseline_high = base_high_reward.clone()
    _baseline_low = base_low_reward.clone()    
    print("FINISHED")
    # clear cache
    torch.cuda.empty_cache()
    
    # variable for evaluation
    global_step = 0
    print("---------------------------------------------")
    print("START TRAINING")
    high_model.train()
    low_model.train()

    for epoch in range(n_epoch):        
        for step in tqdm(range(steps)):    
            # set the optimizer zero gradient
            optimizer.zero_grad()            
            # define state embedding layer
            batch_index = np.random.permutation(n_train_samples)
            batch_index = batch_index[:B]
            X = X_train[batch_index]    

            # ----------------------------------------------------------------------------------------------------------#            
            # define action masks for low and high model
            low_mask = torch.zeros([B, n_cells, n_nodes], dtype= torch.int64).cuda()
            high_mask = torch.zeros([B, n_cells], dtype = torch.int64).cuda()
            # ----------------------------------------------------------------------------------------------------------#
            high_log_prob, low_log_prob, high_cost, low_cost, high_action, low_action = high_model(X, high_mask = high_mask, low_mask = low_mask, low_decoder = low_model)                              

            baseline_high = _baseline_high * beta + high_cost * (1.0 - beta)
            baseline_low = _baseline_low * beta + low_cost * (1.0 - beta)

            # calculate advantage
            high_advantage = high_cost - baseline_high
            low_advantage = low_cost - baseline_low

            # update baseline
            _baseline_high = baseline_high.clone()
            _baseline_low = baseline_low.clone()
            
            # define loss function                    
            high_loss = (high_advantage * high_log_prob).mean()
            low_loss = (low_advantage * low_log_prob).mean()    

            loss = high_loss +  low_loss
            loss.backward()            
                                                
            max_grad_norm = 1.0
            torch.nn.utils.clip_grad_norm_(high_model.parameters(), max_grad_norm, norm_type= 2)
            torch.nn.utils.clip_grad_norm_(low_model.parameters(), max_grad_norm, norm_type= 2)
            optimizer.step()

            low_cost = torch.reshape(low_cost, [B, n_cells])  
            low_cost = torch.sum(low_cost, dim=-1)  
            total_cost = high_cost.mean().cpu() + low_cost.mean().cpu()
            # model evaluation
            high_model.eval()
            low_model.eval()
            if global_step !=0 and global_step % eval_interval == 0:
                print("MODEL EVALUATION")                              
                # generate test valid index
                test_batch_index = np.random.permutation(n_val_samples)
                test_batch_index = test_batch_index[:B]
                
                # get test data from test batch
                test_X = X_val[test_batch_index]

                # ----------------------------------------------------------------------------------------------------------#
                # create a mask for test                                
                test_low_mask = torch.zeros([B, n_cells, n_nodes], dtype= torch.int64).cuda()
                test_high_mask = torch.zeros([B, n_cells], dtype = torch.int64).cuda()
                # ----------------------------------------------------------------------------------------------------------#
                # evaluate the performance                 
                _, _, test_high_cost, test_low_cost, test_high_action, test_low_action = high_model(test_X, high_mask = test_high_mask, low_mask = test_low_mask, low_decoder = low_model)                                  
                test_low_cost = torch.reshape(test_low_cost, [B, n_cells])  
                test_low_cost = torch.sum(test_low_cost, dim=-1)  
                test_total_cost = test_high_cost.mean() + test_low_cost.mean()                
                print("TEST Performance of {}th step:{}".format(global_step, test_total_cost))
                writer.add_scalar("Test Distance", test_total_cost, global_step= global_step)                            

                # visualization                 
                random_batch_index = np.random.randint(B)
                sample = test_X[random_batch_index]
                sample_high_action = test_high_action[random_batch_index]
                sample_low_action = test_low_action[random_batch_index]    
                
                plot_tsp(sample=sample, high_mask=sample_high_action, low_mask=sample_low_action, test_file_name=test_file_name, global_step = global_step)

            # tensorboard 설정 
            if step!=0 and step % log_interval == 0:
                print("SAVE MODEL")
                dir_root = './model/HCPP'
                param_path = dir_root +  "/" + file_name + ".param"
                high_config_path = dir_root + "/" + 'high_' + file_name
                low_config_path = dir_root + "/" + 'low_' + file_name + '.config'

                # make model directory if is not exist
                os.makedirs(dir_root, exist_ok=True)    
                # save high_model            
                torch.save(high_model, high_config_path)                
                # save low_model
                torch.save({
                    'epoch': epoch,
                    'low_model_state_dict': low_model.state_dict()
                }, low_config_path)

                # write information in tensorboard            
                writer.add_scalar("loss", loss, global_step= global_step)
                writer.add_scalar("distance", total_cost, global_step= global_step)

            global_step +=1
            high_model.train()
            low_model.train()

    writer.close()   






