import numpy as np
import matplotlib.pyplot as plt
import pickle
from Decomposition import *
from Boustrophedon_path import *

test_case = "2"
W, H = 70, 70
# robot radius value도 중요한데 이걸 어떻게 설정하면 좋을지 잘 모르겠네
robot_radius = 1

if __name__ == '__main__' :
    for test_case in range(10000):
        print(test_case)
        Boustrophedon_Cellular_Decomposition(test_case)    
        decomposed, total_cells_number, cells = pickle.load(open("./Decomposed data/decomposed_" + str(test_case), "rb"))
        direction_set = [("LEFT", "DOWN"), ("LEFT", "UP"), ("RIGHT", "DOWN"), ("RIGHT", "UP")]
        # cities (start points, end points)
        points = np.zeros([4 * total_cells_number, 2, 2]) # 4 * total_cells_number city, start and end, x and y
        intra_path_length = np.zeros([4 * total_cells_number])
        for cell_id in range(1, total_cells_number + 1) :
            for i in range(4) :
                direction = direction_set[i]
                # cell의 start point와 end point, 그에 따른 path length를 output으로 받음
                start_point, end_point, path_length, flag = Boustrophedon_path(cells[cell_id], direction[0], direction[1], robot_radius)                                        
                if not flag:
                    points[4 * (cell_id - 1) + i] = [start_point, end_point]
                    intra_path_length[4 * (cell_id - 1) + i] = path_length
        # 전체 cell 갯수 중에서 4개 단위로 path length를 cost로 하는 데이터를 생성하면 된다. 
        if not flag:
            # new_points = np.reshape(points, [total_cells_number, 4, 2, 2])
            # new_intra_path_length = np.reshape(intra_path_length, [total_cells_number, 4])
            pickle.dump([points, intra_path_length], open("./Points and costs/PNC_" + str(test_case), "wb"))