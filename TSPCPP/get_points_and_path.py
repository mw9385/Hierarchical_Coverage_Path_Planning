import numpy as np
import matplotlib.pyplot as plt
import pickle
from Decomposition import *
from Boustrophedon_path import *
from visualization import *

# robot radius value도 중요한데 이걸 어떻게 설정하면 좋을지 잘 모르겠네
robot_radius = 1
display = False
count = 0
if __name__ == '__main__' :
    for test_case in range(12800):
        # print(test_case)
        Boustrophedon_Cellular_Decomposition(test_case)    
        decomposed, total_cells_number, cells = pickle.load(open("./Decomposed_data/decomposed_" + str(test_case), "rb"))
        direction_set = [("LEFT", "DOWN"), ("LEFT", "UP"), ("RIGHT", "DOWN"), ("RIGHT", "UP")]
        A, B = np.shape(decomposed)
        # cities (start points, end points)
        points = np.zeros([4 * total_cells_number, 9]) # 4 * total_cells_number city, start, end, path_length
        intra_path_length = np.zeros([4 * total_cells_number])
        internal_path = np.ones([4 * total_cells_number, 200, 2]) * (-1000.0) 
        # display
        if display == True:
            fig = plt.figure()
            plt.imshow(decomposed)
            display_cells(cells, total_cells_number)
        
        none_count = 0
        for cell_id in range(1, total_cells_number + 1) :
            for i in range(4) :
                direction = direction_set[i]
                # cell의 start point와 end point, 그에 따른 path length를 output으로 받음
                start_point, end_point, path_length, flag, path = Boustrophedon_path(cells[cell_id], direction[0], direction[1], robot_radius)                                                                                                        
                if start_point == None or end_point == None or path_length == None:                    
                    none_count +=1  
                else:
                    points[4 * (cell_id - 1) + i, 0:2] = start_point
                    points[4 * (cell_id - 1) + i, 2:4] = end_point
                    points[4 * (cell_id - 1) + i, 4] = len(path)
                    points[4 * (cell_id - 1) + i, 5 + i] = 1 * A # 나중에 130으로 나눠줄꺼라서 일부러 여기서 곱해준다.
                    intra_path_length[4 * (cell_id - 1) + i] = path_length  
                    internal_path[4 * (cell_id - 1) + i, :len(path), :] = path
            plt.show()
        
        # 전체 cell 갯수 중에서 4개 단위로 path length를 cost로 하는 데이터를 생성하면 된다. 
        if none_count ==0:
            pickle.dump([decomposed, total_cells_number, cells], open('./New_maps/decomposed_' + str(count), 'wb'))
            pickle.dump([points, intra_path_length, internal_path], open("./Points_and_costs/PNC_" + str(count), "wb"))
            count +=1
            print('count:', count)
