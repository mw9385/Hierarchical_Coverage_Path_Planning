import copy
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

boundary_color = 'blue'
boundary_width = 2
intra_path_color = 'silver'
inter_path_color = 'dimgray'
show_cell_id = 1

def show_paths(image, num_cells, costs, points, action, path, steps):
    # define size and resize inputs
    width = 3.2    
    image = image.squeeze(0).numpy()   
    image = image* (-1) + 1     
    points = points.squeeze(0)
    action = action.squeeze(0)    
    path = path.squeeze(0)

    plt.figure(figsize=(13, 13))
    plt.imshow(image, cmap='gray')
    # plt.axis('off')
    for i in range(num_cells):
        # index shows the current input and output index
        index = action[i]
        p = points[index]        
        internal_path = path[index]
        
        st = p[0:2]
        g = p[2:4]
        if i == 0:
            plt.plot(st[0], st[1], marker = 'o', color = 'orange', markersize = width + 6.4, zorder = 5, label='Entrance')            
            plt.plot(g[0], g[1], marker = 'X', color = 'springgreen', markersize = width + 6.4, zorder = 5, label = 'Exit') 
            plt.text(st[0]-1, st[1], i, color='blue', fontsize = 20)
        else:
            plt.plot(st[0], st[1], marker = 'o', color = 'orange', markersize = width + 6.4, zorder = 5)
            plt.text(st[0]-1, st[1], i, color='blue', fontsize = 20)
            plt.plot(g[0], g[1], marker = 'X', color = 'springgreen', markersize = width + 6.4, zorder = 5) 
        
        # check the positive points
        internal_index = torch.where(internal_path > 0, 1, 0)  
        internal_index = torch.sum(internal_index, dim=0)
        for k in range(internal_index[0] - 1):
            p1 = internal_path[k]
            p2 = internal_path[k + 1]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color = 'black', linewidth = 1.0, linestyle = '--')
            
        # connect the previous goals and start points        
        if i > 0:
            plt.plot([_g[0], st[0]], [_g[1], st[1]], color='black', linewidth=1.5)
            plt.arrow(_g[0], _g[1], (st[0] - _g[0])*0.9, (st[1] - _g[1])*0.9, width=0.4, color='black')
        # update the previous node
        _g = copy.deepcopy(g)
        plt.pause(0.01)        
        
    plt.show(block=False)
    plt.legend(loc = 'upper right')
    plt.title("Total costs:{}".format(costs[0]))
    plt.savefig('./Results/map_' + str(steps) + '.jpg')
    plt.close('all')

def display_cell(cell, cell_id) :
    # boundary
    
    for y in cell.left :
        plt.plot(cell.min_x, y, marker = 's', color = boundary_color, markersize = boundary_width, zorder = 1)
    
    for y in cell.right :
        plt.plot(cell.max_x, y, marker = 's', color = boundary_color, markersize = boundary_width, zorder = 1)
    
    ceiling_x = sorted(cell.ceiling)
    for i in range(len(ceiling_x)) :
        if i == 0 or abs(cell.ceiling[ceiling_x[i]] - cell.ceiling[ceiling_x[i - 1]]) <= 1 :
            plt.plot(ceiling_x[i], cell.ceiling[ceiling_x[i]], marker = 's', color = boundary_color, markersize = boundary_width, zorder = 1)
        else :
            plt.plot([ceiling_x[i - 1], ceiling_x[i]], [cell.ceiling[ceiling_x[i - 1]], cell.ceiling[ceiling_x[i]]], color = boundary_color, linewidth = boundary_width, zorder = 1)
    
    floor_x = sorted(cell.floor)
    for i in range(len(floor_x)) :
        if i == 0 or abs(cell.floor[floor_x[i]] - cell.floor[floor_x[i - 1]]) <= 1 :
            plt.plot(floor_x[i], cell.floor[floor_x[i]], marker = 's', color = boundary_color, markersize = boundary_width, zorder = 1)
        else :
            plt.plot([floor_x[i - 1], floor_x[i]], [cell.floor[floor_x[i - 1]], cell.floor[floor_x[i]]], color = boundary_color, linewidth = boundary_width, zorder = 1)
    
    if show_cell_id :
        plt.text(cell.center[0] + 0.5, cell.center[1] + 0.5, str(cell_id), color = 'brown', weight = 'bold',
        fontsize = 18, horizontalalignment = 'center', verticalalignment = 'center', zorder = 3)    

def display_cells(cells, total_cells_number) :
    for cell_id in range(1, total_cells_number + 1) :
        cell = cells[cell_id]
        display_cell(cell, cell_id)