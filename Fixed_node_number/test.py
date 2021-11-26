import argparse
import numpy as np
import copy
import torch
import matplotlib.pyplot as plt

from generate_data import TSP
from Attention_model import HCPP, Low_Decoder

# GPU check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device status:{}".format(device))

# main
parser = argparse.ArgumentParser(description="CPP with RL")
parser.add_argument('--n_cells', default=5, help='number of visiting cells')
parser.add_argument('--n_nodes', default=10, help='number of visiting nodes')
parser.add_argument('--max_distance', default=1000, help="maximum distance of nodes from the center of cell")
parser.add_argument('--node_distance', default=1000, help='maximum distance between each cells')
parser.add_argument('--n_hidden', default=128, help="nuber of hidden nodes")
args = vars(parser.parse_args())

B = 128
n_cells = int(args['n_cells'])
n_nodes = int(args['n_nodes'])
n_hidden = int(args['n_hidden'])
max_distance = int(args['max_distance'])
node_distance = int(args['node_distance'])

# generate training data
n_test_samples = 1000
print("---------------------------------------------")
print("GENERATE DATA")
test_tsp_generator = TSP(n_batch=n_test_samples, n_cells = n_cells, n_nodes = n_nodes, max_distance = max_distance, node_distance = node_distance, is_train= False)
X_test = test_tsp_generator.generate_data()
print("FINISHED")

# load model
high_model_path = './model/HCPP/high_HCPP_V1'
low_model_path = './model/HCPP/low_HCPP_V1.config'
low_model = Low_Decoder(n_embedding=n_hidden, n_hidden=n_hidden, n_head=8, C=10).cuda()
low_model.load_state_dict(torch.load(low_model_path)['low_model_state_dict'])
high_model = torch.load(high_model_path).cuda()

def plot_tsp(sample, high_mask, low_mask):    
    
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
    plt.show()    
    # plt.show(block=False)    

def calculate_distance(input_node, output_node):
    return torch.norm(input_node - output_node, dim=0)

if __name__=="__main__":
    # get single sample from the batch
    batch_index = np.random.permutation(n_test_samples)
    batch_index = batch_index[:B]
    X_test = X_test[batch_index]    

    # generate mask         
    low_mask = torch.zeros([B, n_cells, n_nodes], dtype= torch.int64).cuda()    
    high_mask = torch.zeros([B, n_cells], dtype = torch.int64).cuda()     
    _,_, _,_, high_action, low_action  = high_model(X_test, high_mask = high_mask, low_mask = low_mask, low_decoder = low_model)
    """
    [high_action]: batch x n_cells / type: tensor
    [low action]: n_cells x batch x number of local nodes /type: list
    """
    # random selection of batch
    random_batch_index = np.random.randint(B)
    sample = X_test[random_batch_index]
    sample_high_action = high_action[random_batch_index]
    sample_low_action = low_action[random_batch_index]    
    
    plot_tsp(sample=sample, high_mask=sample_high_action, low_mask=sample_low_action)




