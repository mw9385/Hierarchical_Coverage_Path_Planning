import torch
import numpy as np
import matplotlib.pyplot as plt
from Decomposition import Cell

def Boustrophedon_path(cell, horizontal_direction, vertical_direction, robot_radius, plot = False, color = None, width = None) :
    try:
        current_x = min(cell.min_x + robot_radius, cell.max_x)
        finish = False
        path = []
        while not finish :
            min_y = min(cell.floor[current_x] + robot_radius, cell.ceiling[current_x])
            max_y = max(cell.ceiling[current_x] - robot_radius, cell.floor[current_x])
            if vertical_direction == "UP" :
                path.append((current_x, min_y))
                path.append((current_x, max_y))
                vertical_direction = "DOWN"
            else :
                path.append((current_x, max_y))
                path.append((current_x, min_y))
                vertical_direction = "UP"
            if current_x + robot_radius < cell.max_x :
                next_x = min(current_x + robot_radius * 2, cell.max_x)
                for x in range(current_x + 1, next_x) :
                    if vertical_direction == "UP" :
                        y = min(cell.floor[x] + robot_radius, cell.ceiling[x])
                    else :
                        y = max(cell.ceiling[x] - robot_radius, cell.floor[x])
                    path.append((x, y))
                current_x = next_x
            else :
                finish = True
        
        if horizontal_direction == "LEFT" :
            path.reverse()
        
        path_length = 0.0
        for i in range(1, len(path)) :
            A = np.asarray(path[i - 1])
            B = np.asarray(path[i])
            path_length += np.sqrt(np.sum((B - A) ** 2))

        start_point = path[0]
        end_point = path[-1]
        
        flag = False
        path = torch.FloatTensor(path)
        return start_point, end_point, path_length, flag, path
    except:
        print("Exception has been occured.")
        flag = True
        return None, None, None, flag, None
    