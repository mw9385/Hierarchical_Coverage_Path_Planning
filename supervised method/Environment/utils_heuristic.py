import numpy as np
import copy
import time

import torch

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


CONST = 1000.0

def V_V_cost(p, q): # visiting to visiting cost
    return np.sqrt(((p[1] - q[1])**2)+((p[0] - q[0]) **2)) * CONST

def V_C_cost(p,q): # visiting to coverage cost

    assert len(q) == 3, "to_task must be 3 elements (x,y,r)"

    d = V_V_cost(p, q) / CONST
    r = q[-1]

    if d-r < 0: # source point is in the area
        l = r-d
    else: # source point is outside of the area
        l = np.sqrt(d**2 - r**2)

    return (l+7*np.pi*r)* CONST

def V_D_cost(p,q): # visiting to delivery cost

    assert len(q) == 3, "to_task must be 3 elements (x,y,l)"

    d = V_V_cost(p, q) / CONST
    l = q[-1]

    return (d+l) * CONST

def cal_cost(source_task, target_task):

    # Task dependence type
    # 001: coverage, 010: visit, 100: Pick&Place
    if   target_task[-1] == 1: # To Coverage(circle)
        TD_type = "C"
    elif target_task[-2] == 1: # To Visiting
        TD_type = "V"
    elif target_task[-3] == 1: # To Delivery
        TD_type = "D"
    else: # To Depot
        TD_type = "V"

    # from task feature selection
    if   source_task[-1] == 1 or source_task[-2] == 1: # Area or point, ciritical position
        from_task = source_task[:2]
    elif source_task[-3] == 1: # Delivery, place position
        from_task = source_task[3:5]
    else: # Depot
        from_task = source_task[:2]

    # To task feature selection
    if   target_task[-1] == 1 or target_task[-3] == 1: # Area, center position + area_info / Delivery, pick position + area_info
        to_task = target_task[:3]
    elif target_task[-2] == 1: # point, visit position / Delivery, pick position
        to_task = target_task[:2]
    else: # Depot
        to_task = target_task[:2]

    if TD_type == "V":
        cost = V_V_cost(from_task, to_task)
    elif TD_type == "C":
        cost = V_C_cost(from_task, to_task)
    elif TD_type == "D":
        cost = V_D_cost(from_task, to_task)
    else:
        raise NotImplementedError

    return cost

def ORTOOLS_TYPE2(pointset, dist_constraint=None, vehicle_num=None, scale=1000.0, time_verbose=False):
    assert dist_constraint is not None, "Please set distance constraint"
    assert vehicle_num is not None, "Please set the vehicle number initially"
    assert scale == CONST, "check scale value"

    if isinstance(pointset, torch.cuda.FloatTensor) or  isinstance(pointset, torch.FloatTensor):
        pointset = pointset.detach().numpy()

    if time_verbose:
        start = time.time()

    # OUTPUTS
    route_distance = 0
    sol_list = []

    track_idx = np.arange(len(pointset))

    num_points = len(pointset)
    cost_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            if i == j:
                continue
            cost_matrix[i, j] = cal_cost(pointset[i], pointset[j])

    OR_cost_matrix = copy.deepcopy(cost_matrix)

    OR_cost_matrix = OR_cost_matrix.astype(int)
    data = {}
    data['distance_matrix'] = OR_cost_matrix
    data['num_vehicles'] = vehicle_num
    data['depot'] = 0

    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                        data['num_vehicles'],
                                        data['depot'])

    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return cost_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,
        int(dist_constraint),
        True,
        dimension_name
    )


    penalty = 30000
    for node in range(1, len(data['distance_matrix'])):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        for vehicle_id in range(data['num_vehicles']):
            index = routing.Start(vehicle_id)
            sol = []
            while not routing.IsEnd(index):
                sol.append(manager.IndexToNode(index))
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id)

            sol.append(manager.IndexToNode(index)) # Add depot at the last subroutes

            if sol != [0,0]:
                sol_list.append(sol)

    else:
        print("NO SOLUTION FOUND!!")

        assert False, "DEBUGGING REQUIRED"

    if time_verbose:
        return  sol_list, route_distance/scale, time.time()-start
    else:
        return  sol_list, route_distance/scale

def ORTOOLS_TYPE1(pointset, dist_constraint=None, scale=1000.0, time_verbose=False):

    assert dist_constraint is not None, "Please set distance constraint"
    assert scale == CONST, "check scale value"

    if isinstance(pointset, torch.cuda.FloatTensor) or  isinstance(pointset, torch.FloatTensor):
        pointset = pointset.detach().numpy()

    if time_verbose:
        start = time.time()

    # OUTPUTS
    route_distance = 0
    sol_list = []

    track_idx = np.arange(len(pointset))

    while len(pointset) > 1:
        num_points = len(pointset)
        cost_matrix = np.zeros((num_points, num_points))
        for i in range(num_points):
            for j in range(num_points):
                if i == j:
                    continue
                cost_matrix[i, j] = cal_cost(pointset[i], pointset[j])

        OR_cost_matrix = copy.deepcopy(cost_matrix)

        OR_cost_matrix = OR_cost_matrix.astype(int)
        data = {}
        data['distance_matrix'] = OR_cost_matrix
        data['num_vehicles'] = 1
        data['depot'] = 0

        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                            data['num_vehicles'],
                                            data['depot'])

        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return cost_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        dimension_name = 'Distance'
        routing.AddDimension(
            transit_callback_index,
            0,
            int(dist_constraint),
            True,
            dimension_name
        )

        penalty = 30000
        for node in range(1, len(data['distance_matrix'])):
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        # search_parameters.local_search_metaheuristic = (
        #     routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        # search_parameters.time_limit.seconds = 5
        # search_parameters.lns_time_limit.seconds = 5
        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            index = routing.Start(0)
            sol = []
            while not routing.IsEnd(index):
                sol.append(manager.IndexToNode(index))
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, 0)
            sol.append(manager.IndexToNode(index)) # Add depot at the last subroutes

            real_sol = track_idx[sol]

            if list(real_sol) != [0,0]:
                sol_list.append(list(real_sol))

            track_idx = np.delete(track_idx,sol[1:-1])
            pointset = np.delete(pointset,sol[1:-1],0)

        else:
            print("NO SOLUTION FOUND!!")

            assert False, "DEBUGGING REQUIRED"

    if time_verbose:
        return  sol_list, route_distance/scale, time.time()-start
    else:
        return  sol_list, route_distance/scale

def GREEDY(pointset, dist_constraint=None, scale=1000.0, time_verbose=False):

    assert dist_constraint is not None, "Please set distance constraint"

    if isinstance(pointset, torch.cuda.FloatTensor) or  isinstance(pointset, torch.FloatTensor):
        pointset = pointset.detach().numpy()

    if time_verbose:
        start = time.time()

    num_points = len(pointset)
    ret_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            if i == j:
                continue
            ret_matrix[i, j] = cal_cost(pointset[i], pointset[j])

    # ret_matrix = ret_matrix/CONST


    # OUTPUTS
    total_cost = 0
    spend_time = 0
    sol_list = []
    sol =[]

    current_task_idx = 0
    prev_task_idx = 0

    sol.append(current_task_idx)
    cost_to_depot = ret_matrix[:,0]

    visited = np.zeros(len(pointset), dtype=bool)

    while(not visited[1:].all()):
        current_to_task = ret_matrix[current_task_idx]
        total_time = spend_time+current_to_task+cost_to_depot
        exceed_time = total_time > dist_constraint

        mask_constraint = np.logical_or(visited[1:], exceed_time[1:])

        mask_depot = np.logical_and(prev_task_idx==0, (mask_constraint==0).any())

        mask = np.concatenate((mask_depot, mask_constraint),axis=None)

        # Update
        prev_task_idx = current_task_idx
        the_value = min(current_to_task[~mask])
        current_task_idx = np.where(current_to_task==the_value)
        current_task_idx = current_task_idx[0][0]

        sol.append(current_task_idx)
        if current_task_idx == 0 and len(sol)>1:
            sol_list.append(sol)
            sol = []

        total_cost += current_to_task[current_task_idx]
        visited[current_task_idx] = True

        if current_task_idx == 0:
            spend_time = 0
        else:
            spend_time += current_to_task[current_task_idx]

    sol.append(0)
    sol_list.append(sol)

    total_cost += cost_to_depot[current_task_idx]

    if time_verbose:
        return  sol_list, total_cost/scale, time.time()-start
    else:
        return  sol_list, total_cost/scale



# def get_ref_reward(pointset):

#     if isinstance(pointset, torch.cuda.FloatTensor) or  isinstance(pointset, torch.FloatTensor):
#         pointset = pointset.detach().numpy()

#     num_points = len(pointset)
#     ret_matrix = np.zeros((num_points, num_points))
#     for i in range(num_points):
#         for j in range(num_points):
#             if i == j:
#                 continue
#             ret_matrix[i, j] = cal_cost(pointset[i], pointset[j])

#     solver_path = './LKH'

#     problem_instance = tsplib95.models.StandardProblem()
#     problem_instance.type = 'ATSP'
#     problem_instance.dimension = ret_matrix.shape[-1]
#     problem_instance.edge_weight_type = 'EXPLICIT'
#     problem_instance.edge_weight_format = 'FULL_MATRIX'
#     problem_instance.edge_weights = ret_matrix.tolist()

#     q = lkh.solve(solver_path, problem=problem_instance, runs=100, max_candidates="6 symmetric", move_type=3, patching_c=3, patching_a=2, trace_level=0)
#     q = np.array(q[0])-1

#     cost = 0
#     for i in range(num_points):
#         cost += ret_matrix[q[i], q[(i+1) % num_points]]

#     return cost / CONST, ret_matrix / CONST, q
