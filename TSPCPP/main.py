import argparse
import os
# GPU dimension error 발생시 오류가 어디서 생기는지 찾을 수 있는 코드
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import pprint as pp
import json

from tqdm import tqdm
from Decomposition import Cell
from Mission import CPPDataset, my_collate
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from Attention_model import HCPP
from visualization import show_paths

# GPU check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device status:{}".format(device))

# main
parser = argparse.ArgumentParser(description="CPP with RL")
parser.add_argument('--epoch', default= 10000, help="number of epochs")
parser.add_argument('--batch_size', default=128, help="number of batch size")
parser.add_argument('--n_head', default=8, help="number of heads for multi-head attention")
parser.add_argument('--val_size', default=100, help="number of validation samples") # 이게 굳이 필요한가?
parser.add_argument('--beta', type=float, default=0.9, help="beta")
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
parser.add_argument('--n_hidden', default=128, help="nuber of hidden nodes") # 
parser.add_argument('--scaling', type=float, default='130', help='divide the node for scailing')
parser.add_argument('--max_length', type=float, default='200', help='maximum length to suit the different input length')
parser.add_argument('--n_feature', type = int, default= 9, help = 'shows the number of feature')

parser.add_argument('--log_dir', default='./log/V4', type=str, help='directory for the tensorboard')
parser.add_argument('--file_name', default='HCPP_V4', help='file directory')
parser.add_argument('--test_file_name', default='test_performance/V4/', help='test_file directory')
parser.add_argument('--log_interval', default=100, help="store model at every epoch")
parser.add_argument('--eval_interval', default=100, help='update frequency')
args = parser.parse_args()
# save args
with open('./model/model_V4.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)
args = vars(args)

learning_rate = args['lr']
B = int(args['batch_size'])
B_val = int(args['val_size'])
n_epoch = int(args['epoch'])
n_hidden = int(args['n_hidden'])
n_feature = int(args['n_feature'])
n_head = int(args['n_head'])
log_interval = int(args['log_interval'])
eval_interval = int(args['eval_interval'])
file_name = str(args['file_name'])
test_file_name = str(args['test_file_name'])
scaling = int(args['scaling'])
max_length = int(args['max_length'])
beta = float(args['beta'])
# print parameters
pp.pprint(args)

# generate training data
print("---------------------------------------------")
print("GENERATE DATA")
train_generator = CPPDataset(map_dir='New_maps/', pnc_dir='Points_and_costs/', transform=None)
valid_generator = CPPDataset(map_dir='New_maps/', pnc_dir='Points_and_costs/', transform=None)
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
high_model = HCPP(n_feature = n_feature, n_hidden= n_hidden, n_embedding= n_hidden, n_head=n_head, C = 10, scaling=scaling, max_length=max_length).cuda()
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
        baseline_paths = sample_batch[4]
    # get log_prob and reward
    base_high_log_prob, base_high_reward, _ = high_model(baseline_map, baseline_num_cells, baseline_points, baseline_costs)    
    # define initial moving average
    baseline_high = base_high_reward.clone()    
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
            train_path = sample_batch[4]

            high_log_prob, high_cost, high_action = high_model(train_map, train_num_cells, train_points, train_costs)                 
            baseline_high = baseline_high * beta + high_cost * (1.0 - beta)               
            # calculate advantage
            high_advantage = high_cost - baseline_high            

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
            if global_step % eval_interval == 0:
                print("MODEL EVALUATION")                              
                # generate test valid index
                for indicies, sample_batch in zip(range(1), valid_data_loader):    
                    test_map = sample_batch[0]
                    test_num_cells = sample_batch[1]
                    test_points = sample_batch[2]
                    test_costs = sample_batch[3]
                    test_paths = sample_batch[4]

                _, test_high_cost, test_high_action = high_model(test_map, test_num_cells, test_points, test_costs)
                test_total_cost = test_high_cost.mean()
                print("TEST Performance of {}th step:{}".format(global_step, test_total_cost))
                writer.add_scalar("Test Distance", test_total_cost, global_step= global_step)                            

                # visualization
                r_idx = torch.randint(0, B_val, [1])
                s_map = test_map[r_idx]                
                s_num_cells = test_num_cells[r_idx]    
                s_points = test_points[r_idx]            
                s_action = test_high_action[r_idx]                
                s_costs = test_high_cost[r_idx]
                s_paths = test_paths[r_idx]
                show_paths(s_map, s_num_cells, s_costs, s_points, s_action, s_paths, global_step)
            # tensorboard 설정 
            if global_step % log_interval == 0:
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






