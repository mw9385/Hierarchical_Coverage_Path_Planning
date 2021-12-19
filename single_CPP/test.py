import argparse
import numpy as np
import copy
from scipy.ndimage.interpolation import rotate
import torch
import matplotlib.pyplot as plt
from scipy import ndimage

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
# n_test_samples = 1000
# print("---------------------------------------------")
# print("GENERATE DATA")
# test_tsp_generator = TSP(n_batch=n_test_samples, n_nodes = n_nodes, is_train= False)
# X_test = test_tsp_generator.generate_data()
# print("FINISHED")

# load model
high_model_path = './model/HCPP/high_HCPP_V1'
high_model = torch.load(high_model_path).cuda()

def plot_tsp(sample, high_mask, test_file_name, map_order, test_map):    
    # rotated_image = ndimage.rotate(test_map, 270)
    high_mask = high_mask.cpu().numpy()    
    plt.figure()            
    plt.imshow(test_map, cmap='Greys')
    plt.grid()

    R = 0
    # plt.scatter(sample[:, 0], sample[:, 1])
    # for i, high_index in enumerate(high_mask): 
    #     points = sample[high_index] * 20.0
    #     if i == 0:
    #         plt.text(points[0], points[1], 'depot', fontsize=10, label='depot', color='red')
    #     else:
    #         plt.scatter(points[0], points[1], s=10)
    #         plt.plot([_points[0], points[0]], [_points[1], points[1]], color='red')
    #         plt.text(points[0], points[1], i, fontsize = 10)
    #         _reward = calculate_distance(_points, points)
    #         R +=_reward
    #     _points = copy.deepcopy(points)
    #     plt.pause(0.01)
    # plt.title('Total distance:{}'.format(R))    
    # plt.show(block = False)
    # plt.savefig(test_file_name + 'route_' + str(map_order), bbox_inches = 'tight', dpi=1200)
    # plt.close('all')
    
    for i, high_index in enumerate(high_mask): 
        points = sample[high_index] * 20.0
        if i == 0:
            plt.text(points[1], points[0], 'Start', fontsize=10, label='Start', color='red')
        else:
            plt.scatter(points[1], points[0], s=10)
            plt.plot([_points[1], points[1]], [_points[0], points[0]], color='red')
            plt.text(points[1], points[0], i, fontsize = 5, color='white')
            _reward = calculate_distance(_points, points)
            R +=_reward
        _points = copy.deepcopy(points)
        plt.pause(0.01)
    plt.title('Total distance:{}'.format(R))    
    plt.show(block = False)
    plt.savefig(test_file_name + 'route_' + str(map_order), bbox_inches = 'tight', dpi=1200)
    plt.close('all')

def calculate_distance(input_node, output_node):
    return torch.norm(input_node - output_node, dim=0)

if __name__=="__main__":
    # get single sample from the batch
    # load test map
    for i in range(20):
        map_order = i
        test_file_name = './test_performance/mine_mapping/'

        test_map = np.genfromtxt(f'./map/map_idx' + str(map_order) + '.csv', delimiter=',')
        # plt.figure()
        # plt.imshow(test_map)
        # plt.grid()
        # plt.show(block=False)
        # plt.savefig(test_file_name + 'original_' + str(map_order), bbox_inches = 'tight', dpi = 1200)
        # plt.close('all')

        x_position = np.where(test_map==1)[0]
        y_position = np.where(test_map==1)[1]    
        n_points = len(x_position)

        new_position = np.zeros([n_points, 2])
        for i in range(n_points):
            new_position[i, 0]= x_position[i] / 20.0
            new_position[i, 1]= y_position[i] / 20.0
        new_position = torch.from_numpy(new_position).float()
        new_position = new_position.unsqueeze(0)
        
        # batch_index = np.random.permutation(n_test_samples)
        # batch_index = batch_index[:B]
        # X_test = X_test[batch_index]        

        # generate mask             
        # high_mask = torch.zeros([B, n_nodes], dtype = torch.int64).cuda()     
        # high_log_prob, high_reward, high_action  = high_model(X_test, high_mask = high_mask)
        high_mask = torch.zeros([1, n_points], dtype = torch.int64).cuda()         
        high_log_prob, high_reward, high_action  = high_model(new_position, high_mask = high_mask)
        """
        [high_action]: batch x n_cells / type: tensor
        [low action]: n_cells x batch x number of local nodes /type: list
        """    

        # visualization                 
        # random_batch_index = np.random.randint(B)
        # sample = X_test[random_batch_index]
        # sample_high_action = high_action[random_batch_index]     
        # test_file_name = './test_performance/mine_mapping'
        # plot_tsp(sample=sample, high_mask=sample_high_action, test_file_name=test_file_name)

        plot_tsp(sample=new_position.squeeze(0), high_mask=high_action.squeeze(0), test_file_name=test_file_name, map_order = map_order, test_map=test_map)





