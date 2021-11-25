import argparse
import numpy as np
import copy
import torch
import matplotlib.pyplot as plt

from generate_data import TSP
# from Attention_model import HCPP, Low_Decoder

# GPU check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device status:{}".format(device))

# main
parser = argparse.ArgumentParser(description="CPP with RL")
parser.add_argument('--n_nodes', default=10, help='number of visiting nodes')
parser.add_argument('--n_hidden', default=128, help="nuber of hidden nodes")
args = vars(parser.parse_args())

B = 128
n_nodes = int(args['n_nodes'])
n_hidden = int(args['n_hidden'])

# generate training data
n_test_samples = 1000
print("---------------------------------------------")
print("GENERATE DATA")
test_tsp_generator = TSP(n_batch=n_test_samples, n_nodes = n_nodes, is_train= False)
X_test = test_tsp_generator.generate_data()
print("FINISHED")

# load model
# high_model_path = './model/HCPP/high_HCPP_V2'
# high_model = torch.load(high_model_path).cuda()

def plot_tsp(sample):    
    
    plt.figure()        
    plt.grid()
    plt.scatter(sample[:, 0], sample[:, 1], s= 30)        
    plt.show()
    plt.close()

def calculate_distance(input_node, output_node):
    return torch.norm(input_node - output_node, dim=0)

if __name__=="__main__":
    # # get single sample from the batch
    batch_index = np.random.permutation(n_test_samples)
    batch_index = batch_index[:B]
    X_test = X_test[batch_index]        

    # generate mask             
    # high_mask = torch.zeros([B, n_nodes], dtype = torch.int64).cuda()     
    # high_log_prob, high_reward, high_action  = high_model(X_test, high_mask = high_mask)
    """
    [high_action]: batch x n_cells / type: tensor
    [low action]: n_cells x batch x number of local nodes /type: list
    """
    # random selection of batch
    random_batch_index = np.random.randint(B)
    sample = X_test[random_batch_index]      
    plot_tsp(sample=sample)




