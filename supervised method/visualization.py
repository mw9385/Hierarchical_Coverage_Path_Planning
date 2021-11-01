"""
Plot result from the test data
"""
import argparse
from numpy.lib.index_tricks import _ix__dispatcher
from numpy.random import seed
from ortools.constraint_solver.pywrapcp import RoutingModel
from IPython.display import clear_output
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from solver import solver_Attention, solver_RNN
from Environment import Mission
from Environment.utils_heuristic import ORTOOLS_TYPE1, ORTOOLS_TYPE2
from Environment.utils import Dae_Bup_Gwan
from Environment import State
from Model_Loader import load_model_eval
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


colors = mcolors.TABLEAU_COLORS
color_name = list(mcolors.TABLEAU_COLORS)


parser = argparse.ArgumentParser()

parser.add_argument("--model_type", type=str, default="att")
parser.add_argument("--coverage_num", type=int, default=4)
parser.add_argument("--visiting_num", type=int, default=4)
parser.add_argument("--pick_place_num", type=int, default=4)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--num_tr_dataset", type=int, default=10000)
parser.add_argument("--num_te_dataset", type=int, default=2000)
parser.add_argument("--embedding_size", type=int, default=128)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--grad_clip", type=float, default=1.5)
parser.add_argument("--use_cuda", type=bool, default=True)
parser.add_argument("--beta", type=float, default=0.9)
args = parser.parse_args()

num_sample = 1

if args.use_cuda:
  use_pin_memory = True
else:
  use_pin_memory = False

model_path_list = []
model_path_list.append("/home/mw/machine_learning/Heterogeneous_Task_Routing/models/V3/pointerNet.param")
print("wow")

for load_path in model_path_list:
  model = load_model_eval(load_path)
  model = model.cuda()


model.eval()
test_dataset = Mission.MissionDataset(num_visit=args.visiting_num, num_coverage=args.coverage_num, num_pick_place=args.pick_place_num,
                                      num_samples=num_sample,
                                      random_seed=37,
                                      overlap=False)

test_data_loader = DataLoader(
  test_dataset,
  batch_size = num_sample,
  shuffle=True,
  pin_memory=use_pin_memory)

def Route_to_subroutes(solution):
  ## Devide solution in to the sub routes
    routes = []
    sub_route = []
    double_check = 0
    for sol in solution[1:]:
      if sol == 0:
        double_check += 1
        if double_check == 2:
          break
        sub_route.insert(0,0)
        routes.append(sub_route)
        sub_route=[]
      else:
        double_check = 0
        sub_route.append(sol)
    if len(sub_route)>0:
      sub_route.insert(0,0)
      routes.append(sub_route)
    return routes


for i, mission in test_data_loader:
  if args.use_cuda:
      mission = mission.cuda()

log_prob, _, solution, cost,_  = model(mission)
print(solution)
# mission2 = mission.clone().cpu()
mission = mission[0].cpu().numpy()

Depot = mission[:1, :]
Area = mission[1:args.coverage_num+1, :]
Visit = mission[args.coverage_num+1:, :]
solution = solution.cpu().numpy()[0]
routes = Route_to_subroutes(solution)

print("==============MY SOLUTION================")
print("My solution:", routes)
print("Cost:", cost.item())

SCALE = 1000
DISTANCE = 6.0
ortools_type1_solution, _ = ORTOOLS_TYPE1(mission, dist_constraint=DISTANCE*SCALE, scale=SCALE)
ortools_type1_cost, _ = Dae_Bup_Gwan(mission, ortools_type1_solution)

print("==============OR TOOLS Type1================")
print("ORTOOLS solution:",ortools_type1_solution)
print("Cost:",ortools_type1_cost)

ortools_type2_solution, _ = ORTOOLS_TYPE2(mission, dist_constraint=DISTANCE*SCALE,vehicle_num=len(mission),scale=SCALE)
ortools_type2_cost, _ = Dae_Bup_Gwan(mission, ortools_type2_solution)

print("==============OR TOOLS Type2================")
print("ORTOOLS solution:",ortools_type2_solution)
print("Cost:",ortools_type2_cost)



###########################################################
# Plot
###########################################################
def contact_point(point, center, radius):
  assert isinstance(point, tuple), "point must be tuple"
  assert isinstance(center, tuple), "center must be tuple"

  point = np.array(point)
  center = np.array(center)

  d = np.linalg.norm(point-center)

  if d <= radius:
    vector = (center-point)/d
    heading = - (radius-d) * vector
    contact_point = point + heading

  else:
    h = abs(center[1]-point[1])
    l = (d**2 - radius**2)**0.5
    theta = math.asin(h/d)
    beta = math.acos(l/d)
    alpha = theta-beta

    dx = l * math.cos(alpha)
    dy = l * math.sin(alpha)

    x_sign = 1 if center[0] > point[0] else -1
    y_sign = 1 if center[1] > point[1] else -1

    contact_point = point + np.array((x_sign * dx, y_sign * dy))

  return contact_point

fit, ax = plt.subplots(2,2, figsize=(6,6))
ax[0,0].set_title("%d task(%dV/%dC/%dD)" % (args.visiting_num+args.coverage_num+args.pick_place_num,args.visiting_num, args.coverage_num, args.pick_place_num))
ax[0,1].set_title("My Solution: %.3f" %(cost))
ax[1,0].set_title("ORTOOLS Type1: %.3f" %(ortools_type1_cost))
ax[1,1].set_title("ORTOOLS Type2: %.3f" %(ortools_type2_cost))


ax[0,0].scatter(Depot[0][:1], Depot[0][1:2],marker='s', c='k', s=30, label='Depot')
ax[0,1].scatter(Depot[0][:1], Depot[0][1:2],marker='s', c='k', s=30, label='Depot')
ax[1,0].scatter(Depot[0][:1], Depot[0][1:2],marker='s', c='k', s=30, label='Depot')
ax[1,1].scatter(Depot[0][:1], Depot[0][1:2],marker='s', c='k', s=30, label='Depot')

theta = np.radians(np.linspace(0,360*5,1000))


# Plot Mission
for i in range(1,len(mission)):
  task = mission[i]

  if task[-2] == 1:
    point = task[:2]

    ax[0,0].scatter(point[:1], point[1:2],marker='s', color='b', s=10, label='Visiting')
    ax[0,1].scatter(point[:1], point[1:2],marker='s', color='b', s=10, label='Visiting')
    ax[1,0].scatter(point[:1], point[1:2],marker='s', color='b', s=10, label='Visiting')
    ax[1,1].scatter(point[:1], point[1:2],marker='s', color='b', s=10, label='Visiting')

  elif task[-1] == 1:
    x, y, r = task[:3]

    ax[0,0].add_patch(plt.Circle((x, y), r, fill=False))
    ax[0,1].add_patch(plt.Circle((x, y), r, fill=False))
    ax[1,0].add_patch(plt.Circle((x, y), r, fill=False))
    ax[1,1].add_patch(plt.Circle((x, y), r, fill=False))

  elif task[-3] == 1:
    pick_point = task[:2]
    place_point = task[3:5]
    points = np.concatenate((pick_point[None,:], place_point[None,:]), axis=0)

    ax[0,0].scatter(points[:,0], points[:,1], marker='D', color='m', s=20)
    ax[0,1].scatter(points[:,0], points[:,1], marker='D', color='m', s=20)
    ax[1,0].scatter(points[:,0], points[:,1], marker='D', color='m', s=20)
    ax[1,1].scatter(points[:,0], points[:,1], marker='D', color='m', s=20)

    ax[0,0].arrow(pick_point[0], pick_point[1], 0.8*(place_point[0]-pick_point[0]), 0.8*(place_point[1]-pick_point[1]), width=0.002, color='c', head_width=0.012)
    ax[0,1].arrow(pick_point[0], pick_point[1], 0.8*(place_point[0]-pick_point[0]), 0.8*(place_point[1]-pick_point[1]), width=0.002, color='c', head_width=0.012)
    ax[1,0].arrow(pick_point[0], pick_point[1], 0.8*(place_point[0]-pick_point[0]), 0.8*(place_point[1]-pick_point[1]), width=0.002, color='c', head_width=0.012)
    ax[1,1].arrow(pick_point[0], pick_point[1], 0.8*(place_point[0]-pick_point[0]), 0.8*(place_point[1]-pick_point[1]), width=0.002, color='c', head_width=0.012)

# Plot SOLUTION
colors = mcolors.TABLEAU_COLORS
color_name = list(mcolors.TABLEAU_COLORS)

ids = 0

for sub_route in routes:
  prev = Depot[0]
  for i in sub_route[1:]:
    task = mission[i]

    if task[-2] == 1:
      point = task[:2]
      ax[0,1].plot([prev[0], point[0]], [prev[1], point[1]], color=colors[color_name[ids]], linewidth=0.5)

      prev = point

    elif task[-1] == 1:
      x, y, r = task[:3]

      spiral_r = theta / 31 * r
      spiral_x = spiral_r*np.cos(theta)+x
      spiral_y = spiral_r*np.sin(theta)+y
      ax[0,1].plot(spiral_x, spiral_y, color=colors[color_name[ids]], linewidth=0.5)

      contact = contact_point((prev[0], prev[1]),(x,y),r)
      ax[0,1].plot([prev[0], contact[0]], [prev[1], contact[1]], color=colors[color_name[ids]], linewidth=0.5)

      prev = np.array([x, y])

    elif task[-3] == 1:
      pick_point = task[:2]
      place_point = task[3:5]
      points = np.concatenate((pick_point[None,:], place_point[None,:]), axis=0)

      ax[0,1].plot([prev[0], pick_point[0]], [prev[1], pick_point[1]], color=colors[color_name[ids]], linewidth=0.5)

      prev = place_point

  ax[0,1].plot([prev[0],0],[prev[1],0],'k--',linewidth=0.5, label='Last path')
  ids += 1

custom_lines = [Line2D([0], [0], color=color_name[i], lw=1) for i in range(ids)]
# ax[0,1].legend(custom_lines, ["Route %d"%(i+1) for i in range(len(custom_lines))], loc='upper left')

# Plot ORTOOLS_TYPE1 route
colors = mcolors.TABLEAU_COLORS
color_name = list(mcolors.TABLEAU_COLORS)
ids = 0

for sub_route in ortools_type1_solution:
  prev = Depot[0]
  for i in sub_route[1:-1]:
    task = mission[i]

    if task[-2] == 1:
      point = task[:2]
      ax[1,0].plot([prev[0], point[0]], [prev[1], point[1]], color=colors[color_name[ids]], linewidth=0.5)

      prev = point

    elif task[-1] == 1:
      x, y, r = task[:3]

      spiral_r = theta / 31 * r
      spiral_x = spiral_r*np.cos(theta)+x
      spiral_y = spiral_r*np.sin(theta)+y
      ax[1,0].plot(spiral_x, spiral_y, color=colors[color_name[ids]], linewidth=0.5)

      contact = contact_point((prev[0], prev[1]),(x,y),r)
      ax[1,0].plot([prev[0], contact[0]], [prev[1], contact[1]], color=colors[color_name[ids]], linewidth=0.5)

      prev = np.array([x, y])

    elif task[-3] == 1:
      pick_point = task[:2]
      place_point = task[3:5]
      points = np.concatenate((pick_point[None,:], place_point[None,:]), axis=0)

      ax[1,0].plot([prev[0], pick_point[0]], [prev[1], pick_point[1]], color=colors[color_name[ids]], linewidth=0.5)

      prev = place_point

  ax[1,0].plot([prev[0],0],[prev[1],0],'k--',linewidth=0.5, label='Last path')
  ids += 1

custom_lines = [Line2D([0], [0], color=color_name[i], lw=1) for i in range(ids)]
# ax[1,0].legend(custom_lines, ["Route %d"%(i+1) for i in range(len(custom_lines))], loc='upper left')

# Plot ORTOOLS_TYPE2 route
colors = mcolors.TABLEAU_COLORS
color_name = list(mcolors.TABLEAU_COLORS)
ids = 0

for sub_route in ortools_type2_solution:
  prev = Depot[0]
  for i in sub_route[1:-1]:
    task = mission[i]

    if task[-2] == 1:
      point = task[:2]
      ax[1,1].plot([prev[0], point[0]], [prev[1], point[1]], color=colors[color_name[ids]], linewidth=0.5)

      prev = point

    elif task[-1] == 1:
      x, y, r = task[:3]

      spiral_r = theta / 31 * r
      spiral_x = spiral_r*np.cos(theta)+x
      spiral_y = spiral_r*np.sin(theta)+y
      ax[1,1].plot(spiral_x, spiral_y, color=colors[color_name[ids]], linewidth=0.5)

      contact = contact_point((prev[0], prev[1]),(x,y),r)
      ax[1,1].plot([prev[0], contact[0]], [prev[1], contact[1]], color=colors[color_name[ids]], linewidth=0.5)

      prev = np.array([x, y])

    elif task[-3] == 1:
      pick_point = task[:2]
      place_point = task[3:5]
      points = np.concatenate((pick_point[None,:], place_point[None,:]), axis=0)

      ax[1,1].plot([prev[0], pick_point[0]], [prev[1], pick_point[1]], color=colors[color_name[ids]], linewidth=0.5)

      prev = place_point

  ax[1,1].plot([prev[0],0],[prev[1],0],'k--',linewidth=0.5, label='Last path')
  ids += 1

custom_lines = [Line2D([0], [0], color=color_name[i], lw=1) for i in range(ids)]
# ax[1,1].legend(custom_lines, ["Route %d"%(i+1) for i in range(len(custom_lines))], loc='upper left')


ax[0,0].set_xlim((-0.05, 1.1))
ax[0,0].set_ylim((-0.05, 1.1))
ax[0,0].set_aspect('equal')

ax[0,1].set_xlim((-0.05, 1.1))
ax[0,1].set_ylim((-0.05, 1.1))
ax[0,1].set_aspect('equal')

ax[1,0].set_xlim((-0.05, 1.1))
ax[1,0].set_ylim((-0.05, 1.1))
ax[1,0].set_aspect('equal')

ax[1,1].set_xlim((-0.05, 1.1))
ax[1,1].set_ylim((-0.05, 1.1))
ax[1,1].set_aspect('equal')

# plt.tight_layout()
plt.show()