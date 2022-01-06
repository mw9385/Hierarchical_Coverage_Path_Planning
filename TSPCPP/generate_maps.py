import numpy as np
import math
import matplotlib.pyplot as plt
import gym

from gym import spaces
from gym.utils import seeding
from grid_map_lib import GridMap

class Env(gym.Env):
    """Custom Environment that follows gym interface"""
    def __init__(self, width, height):
        super(Env, self).__init__()
        self.width = width
        self.height = height
        self.action_space = spaces.Discrete(4)                
        self.observation_space = spaces.Dict({            
            'map_info': spaces.Box(low=0.0, high=1.0, shape=(self.width*self.height,), dtype = np.float32) # sensor 정보            
        })
        self.direction = [np.array((-1,0)), np.array((1,0)), np.array((0, -1)), np.array((0,1))] # agent가 움직이는 방향을 설정해준다.                                 
                
        # initialize drones        
        self.start_position = [5, 5]
        
        # initial setup for the map
        ox = [0.0, 5.0, 10.0, 15.0, 20.0, 15.0, 0.0]
        oy = [0.0, -5.0, 0.0, 5.0, 10.0, 20.0, 0.0]
        
        # define observation map
        self.grid_map = self.setup_grid_map(ox, oy, resolution = 1.0)
        self.grid_data = self.grid_map.get_grid_map()                
        self.grid_data[self.start_position[0]][self.start_position[1]] = 2
        
        # initial conditions       
        self.seed()

    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def setup_grid_map(self, ox, oy, resolution, offset_grid= 20):        
        width = math.ceil((max(ox) - min(ox)) / resolution) + offset_grid
        height = math.ceil((max(oy) - min(oy)) / resolution) + offset_grid
        self.center_x = (np.max(ox) + np.min(ox)) / 2.0
        self.center_y = (np.max(oy) + np.min(oy)) / 2.0        

        grid_map = GridMap(self.width, self.height, resolution, self.center_x, self.center_y)
        grid_map.set_value_from_polygon(ox, oy, 1.0, inside=False)
        grid_map.expand_grid()            
        return grid_map
    
    def step(self, action):
        # take action
        self.next_position = self.start_position + self.direction[action]

        # Exception when the agent encounters the boundary of the map.        
        self.bad_action = False
        if action == 0: # move upward
            if self.next_position[0] < 0:
                self.next_position[0] = self.next_position[0] + 1
                self.bad_action = True
        elif action == 1: # move down
            if self.next_position[0] > self.width - 1:
                self.next_position[0] = self.next_position[0] - 1
                self.bad_action = True
        elif action == 2: # move left
            if self.next_position[1] < 0:
                self.next_position[1] = self.next_position[1] + 1
                self.bad_action = True
        elif action == 3: # move right
            if self.next_position[1] > self.height - 1:
                self.next_position[1] = self.next_position[1] - 1 
                self.bad_action = True

        # define reward
        reward = 0.0 # initial reward        
        if self.grid_data[self.next_position[0]][self.next_position[1]] == 1: # unexplored cell        
            reward = 2.0
        else:
            reward += 0.0                    
        
        # update start_position
        self.start_position = self.next_position

        # define booliean function
        # all cells are explored except the agent cell
        clear_done = bool(np.sum(self.grid_data) == 2) 
        crash_done = bool(self.bad_action)

        if clear_done:
            reward += 10.0
        else:
            reward += 0.0                        
        
        if crash_done:
            reward -= 0.0
        
        done = bool(clear_done or crash_done)
        info = {}       
        
        # define the state
        map_info = self.grid_data.flatten()                           
        return dict(position= self.next_position, map_info = map_info), reward, done, info

    def reset(self):
        # 임의의 polygon을 다시 생성         
        rand_x = self.np_random.randint(self.width)
        if 0 <= rand_x <= 10 or self.width - 10 <= rand_x <= self.width -1:
            rand_y = self.np_random.randint(self.height - 1)
        else:
            rand_y = self.np_random.randint(5)            

        self.start_position = np.array([rand_x, rand_y])        
        rand_int = self.np_random.randint(3)
        if rand_int == 0:
            rand_ox = self.np_random.randint(-10, 10, 7)
            rand_oy = self.np_random.randint(-10, 10, 7)
            ox = np.array([0.0, 20.0, 40.0, 90.0, 100.0, 40.0, 0.0]) + rand_ox
            oy = np.array([0.0, -20.0, 0.0, 30.0, 60.0, 80.0, 0.0]) + rand_oy
            ox = ox.tolist()
            oy = oy.tolist()
        elif rand_int == 1:
            rand_ox = self.np_random.randint(-10, 10, 5)
            rand_oy = self.np_random.randint(-10, 10, 5)
            ox = np.array([0.0, 50.0, 50.0, 0.0, 0.0]) + rand_ox
            oy = np.array([0.0, 0.0, 30.0, 30.0, 0.0]) + rand_oy
            ox = ox.tolist()
            oy = oy.tolist()
        else:
            rand_ox = self.np_random.randint(-10, 10, 7)
            rand_oy = self.np_random.randint(-10, 10, 7)
            ox = np.array([0.0, 20.0, 50.0, 80.0, 70.0, 40.0, 0.0]) + rand_ox
            oy = np.array([0.0, -80.0, 0.0, 30.0, 60.0, 80.0, 0.0]) + rand_oy
            ox = ox.tolist()
            oy = oy.tolist()
        
        self.grid_map = self.setup_grid_map(ox, oy, resolution = 1.0)
        self.grid_data = self.grid_map.get_grid_map()        
        map_info = self.grid_data.flatten()        
        return dict(position = self.start_position, map_info = map_info)        

    def render(self, mode = 'human'):
        img = self.grid_data   
        return img

if __name__=="__main__":    
    env = Env(width = 130, height = 130)
    env.seed(100)    
    fig = plt.figure(figsize = (10,8))    
    e = 0    

    for j in range(12800):
        observation = env.reset()  
        for step in range(1):        
            temp = observation['map_info']
            temp = temp * (-1) + 1
            temp = np.reshape(temp, [130, 130])
            # turn off the displaying
            plt.ioff()
            plt.imshow(temp, cmap=plt.cm.gray)
            plt.axis('off')
            # plt.show(block=False)    

            # save figure 
            plt.savefig('./Training maps/map_' + str(j) + '.jpg', bbox_inches = 'tight', dpi=600)            
            plt.close('all')

                
        

