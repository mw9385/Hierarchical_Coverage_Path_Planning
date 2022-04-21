import argparse
import numpy as np
import random
import tsplib95

from scipy.spatial import distance_matrix
from optimizer import optimized
from time import time

parser = argparse.ArgumentParser(description='tsp')
parser.add_argument('--num_tasks', type=int, default=100, help='number of tasks')
parser.add_argument('--scale', type=int, default=10000, help='scale factor for distance matrix')
parser.add_argument('--file_dir', type=str, default='./SAMPLES/')
parser.add_argument('--save_file_dir', type=str, default='./SOLUTIONS/')
parser.add_argument('--file_name', type=str, default='test')
parser.add_argument('--pass_range', type=int, default=3, help = 'maximum number of times an agent may have to revisit a node')
parser.add_argument('--num_agents', type=int, default=5, help='number of agents')
args = parser.parse_args()

identifier = str(time())

def get_ref_reward(pointset, name, args):
    pointset = pointset
    solution_matrix = distance_matrix(pointset, pointset)    
    solution_matrix = solution_matrix.astype(int)

    problem_instance = tsplib95.models.StandardProblem()    
    problem_instance.name = name
    problem_instance.type = 'TSP'
    problem_instance.dimension = solution_matrix.shape[-1]
    problem_instance.edge_weight_type = 'EXPLICIT'
    problem_instance.edge_weight_format = 'FULL_MATRIX'
    problem_instance.edge_weights = solution_matrix.tolist()
    
    # save problem
    problem_instance.save(args.file_dir + str(name))
    pass_count = [random.randint(1, max(1, args.pass_range)) for _ in range(args.num_tasks)]    
    opt_cost, opt_path = optimized.LKH(
        solution_matrix, pass_count= pass_count,
        num_agents= args.num_agents,
        identifier= identifier, is_tour = True)

    cost = []
    for costs in opt_cost:
        cost.append(costs/args.scale)

    print('opt_cost:{}'.format(cost))
    print('opt_path:{}'.format(opt_path))
    
    # save solution and costs in text format
    f = open("{}{}.tour".format(args.save_file_dir, name), 'w')
    f.write("OPTIMAL_COST = {} \n \
    OPTIMAL_PATH= {}".format(cost, opt_path))

def main(args):    
    pointset = np.random.uniform(-1, 1, size = (args.num_tasks, 2))*args.scale
    name = 1
    get_ref_reward(pointset, name, args)    

if __name__=='__main__':
    main(args)    




    
