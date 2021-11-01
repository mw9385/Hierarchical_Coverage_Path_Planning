from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import math

from Environment import Mission


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


def task_plot(tasks, args):
  fig, ax = plt.subplots()
  ax.set_title("%d visiting, %d coverage, %d delivery" % (args.visiting_num, args.coverage_num, args.pick_place_num))

  # Start_point, Area_Info, End_point [x1,y1,a,x2,y2]
  Depot           = tasks[0]
  coverage_task   = tasks[1:args.coverage_num+1, :]
  visit_task      = tasks[args.coverage_num+1:args.coverage_num+args.visiting_num+1, :]
  pick_place_task = tasks[args.coverage_num+args.visiting_num+1:, :]

  # Depot
  ax.scatter(Depot[:1], Depot[1:2], marker='s', color='k', s=30)

  # Area
  for i in range(args.coverage_num):
    x, y, r, _, _, _, _ ,_ = coverage_task[i]
    circle = plt.Circle((x, y), r, fill=False)
    ax.add_patch(circle)

  # visit
  ax.scatter(visit_task[:, 0], visit_task[:, 1], marker='s', color='b', s=10)

  # Pick Place
  pick_point = pick_place_task[:,:2]
  place_point = pick_place_task[:,3:5]
  points = np.concatenate((pick_point, place_point), axis=0)

  ax.scatter(points[:,0], points[:,1], marker='D', color='m', s=20)
  for i in range(args.pick_place_num):
    ax.arrow(pick_point[i][0], pick_point[i][1], 0.8*(place_point[i][0]-pick_point[i][0]), 0.8*(place_point[i][1]-pick_point[i][1]), width=0.002, color='c', head_width=0.012)


  ax.set_xlim((-0.05, 1.2))
  ax.set_ylim((-0.05, 1.2))
  ax.set_aspect('equal')
  plt.show()


def test(args):
    train_loader = DataLoader(Mission.MissionDataset(args.visiting_num, args.coverage_num, args.pick_place_num,
                                                     num_samples=1, random_seed=16, overlap=False),
                              batch_size=1, shuffle=True, num_workers=1)

    for  (batch_idx, task_list_batch) in train_loader:
        task_plot(task_list_batch[0].cpu().numpy(), args)

if __name__=="__main__":


  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--coverage_num", type=int, default=5)
  parser.add_argument("--visiting_num", type=int, default=5)
  parser.add_argument("--pick_place_num", type=int, default=5)
  args = parser.parse_args()

  test(args)
