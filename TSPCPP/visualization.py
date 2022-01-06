import copy
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_paths(image, num_cells, costs, points, action, steps):
    # define size and resize inputs
    width = 3.2    
    image = image.squeeze(0).numpy()   
    image = image* (-1) + 1     
    points = points.squeeze(0)
    action = action.squeeze(0)
    
    plt.figure(figsize=(13, 13))
    plt.imshow(image, cmap='gray')
    # plt.axis('off')
    for i in range(num_cells):
        # index shows the current input and output index
        index = action[i]
        p = points[index]
        st = p[0]
        g = p[1]
        if i == 0:
            plt.plot(st[0], st[1], marker = 'o', color = 'orange', markersize = width + 6.4, zorder = 5, label='Entrance')            
            plt.plot(g[0], g[1], marker = 'X', color = 'springgreen', markersize = width + 6.4, zorder = 5, label = 'Exit') 
            plt.text(st[0]-1, st[1], i, color='blue', fontsize = 20)
        else:
            plt.plot(st[0], st[1], marker = 'o', color = 'orange', markersize = width + 6.4, zorder = 5)
            plt.text(st[0]-1, st[1], i, color='blue', fontsize = 20)
            plt.plot(g[0], g[1], marker = 'X', color = 'springgreen', markersize = width + 6.4, zorder = 5) 
        # connect the previous goals and start points        
        if i > 0:
            plt.plot([_g[0], st[0]], [_g[1], st[1]], color='red', linewidth=1.5)
            plt.arrow(_g[0], _g[1], (st[0] - _g[0])*0.9, (st[1] - _g[1])*0.9, width=0.4, color='red')
        # update the previous node
        _g = copy.deepcopy(g)
        plt.pause(0.01)        
        
    plt.show(block=False)
    plt.legend(loc = 'upper right')
    plt.title("Total costs:{}".format(costs[0]))
    plt.savefig('./Results/map_' + str(steps) + '.jpg')
    plt.close('all')
