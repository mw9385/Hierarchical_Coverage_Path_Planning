import torch
from typing import NamedTuple

from Environment.utils import cal_time_budget_many2many, Batch_cal_cost
# For TEST
# from utils import cal_time_budget_many2many, Batch_cal_cost


class TA_State(NamedTuple):
  # mission state
  mission: torch.Tensor
  task_num: int

  # vehicle state
  prev_task: torch.Tensor
  spend_time: torch.Tensor
  visited: torch.Tensor
  cur_task: torch.Tensor
  step: torch.Tensor

  # constraint information
  VEHICLE_TIME_ABILITY = 6.0
  cost2depot: torch.Tensor

  # util variable
  ids: torch.Tensor

  def __getitem__(self, key):
    pass

  @staticmethod
  def initialize(inputs, visited_dtype=torch.uint8) -> NamedTuple:
    """
    [Input]
      inputs: batch_size x seq_len x feature_size
    [Return]
      state: NamedTuple
    """
    batch_size = inputs.shape[0]

    mission=inputs
    task_num = inputs.shape[1]

    prev_task=torch.zeros(batch_size, 1, dtype=torch.long, device=inputs.device)
    spend_time=torch.zeros(batch_size, 1, device=inputs.device)
    visited=torch.zeros(batch_size,task_num,1,dtype=torch.uint8, device=inputs.device)
    cur_task=inputs[:,0,:][:,None,:] # batch x 1 x feature_size
    step=torch.zeros(1, dtype=torch.int64, device=inputs.device)

    cost2depot=cal_time_budget_many2many(mission, cur_task.repeat(1,task_num,1)) # precalculate all tasks to depot cost

    assert ((cost2depot*2)<6.0).all(), "Minimum required fuel is higher than configureation"

    ids = torch.arange(batch_size, dtype=torch.int64, device=inputs.device)[:, None]

    return TA_State(
      mission=mission,
      task_num=task_num,
      prev_task=prev_task,
      spend_time=spend_time,
      visited=visited,
      cur_task=cur_task,
      step=step,
      cost2depot=cost2depot,
      ids=ids
    )

  def all_finished(self):
    """
    Every tasks are done except depot
    """
    return (self.visited[:,1:,:]).all()

  def update(self, selected):
    """
    [Input]
      selected: batch
    [Return]
      replace updated information to the state
    """

    assert self.step.size(0) == 1, "Can only update if state represents single step"

    selected = selected[:, None]

    # prev_task = selected

    selected_task = self.mission[self.ids, selected] # batch x 1 x feature_size

    spend_time = self.spend_time + Batch_cal_cost(self.cur_task.squeeze(1), selected_task.squeeze(1), selected.shape[0])[:,None] # batch x 1
    spend_time = spend_time * (selected != 0).float()

    visited = self.visited.scatter(1, selected[:, :, None], 1) # batch x task_num x 1

    return self._replace(
      prev_task = selected, spend_time=spend_time, visited=visited, cur_task = selected_task, step=self.step+1
    )

  def get_remained_time(self):
    """
    Return Rmained time budget
    [Return]
      remained_time: batch_size x 1
    """

    remained_time = (self.VEHICLE_TIME_ABILITY - self.spend_time)/self.VEHICLE_TIME_ABILITY
    return remained_time

  def get_mask(self):
    """
    Forbids to visit depot twice in a row, unless all nodes have been visited
    [Return]
      mask: batch_size X task_num, dtype=torch.bool
    """


    visited_task = self.visited[:,1:,:] # batch x task_num-1 x 1

    current_to_tasks=cal_time_budget_many2many(self.cur_task.repeat(1,self.task_num,1), self.mission)
    total_time = ( self.spend_time[:,:,None]
                   + cal_time_budget_many2many(self.cur_task.repeat(1,self.task_num,1), self.mission)
                   + self.cost2depot)

    exceeds_time = total_time > self.VEHICLE_TIME_ABILITY # batch x task_num x 1

    # exceeds_time = ( self.spend_time[:,:,None]
    #                + cal_time_budget_many2many(self.cur_task.repeat(1,self.task_num,1), self.mission)
    #                + self.cost2depot) > self.VEHICLE_TIME_ABILITY # batch x task_num x 1

    # print("Spend time")
    # print(self.spend_time)
    # print("TIME of cur to all")
    # print(cal_time_budget_many2many(self.cur_task.repeat(1,self.task_num,1), self.mission).squeeze(-1))
    # print("TIME of all to depot")
    # print(self.cost2depot.squeeze(-1))
    # print("SAFETY TIME")
    # print((cal_time_budget_many2many(self.cur_task.repeat(1,self.task_num,1), self.mission) + self.cost2depot).squeeze(-1))

    exceeds_time = exceeds_time[:,1:,:] # batch x task_num-1 x 1

    mask_constraint = torch.logical_or(visited_task, exceeds_time) # batch x task_num-1 x 1

    mask_depot = torch.logical_and(self.prev_task == 0, (mask_constraint==0).any(1)) # batch x 1

    mask = torch.cat((mask_depot, mask_constraint.squeeze(-1)),dim=-1) # batch x task_num

    return mask

if __name__ == "__main__":
    #
    from torch.utils.data import DataLoader, Dataset
    import Mission
    from tqdm import tqdm

    import argparse
    from State import TA_State


    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage_num", type=int, default=2)
    parser.add_argument("--visiting_num", type=int, default=2)
    parser.add_argument("--pick_place_num", type=int, default=2)
    args = parser.parse_args()

    sample_num = 10
    train_loader = DataLoader(Mission.MissionDataset(args.visiting_num, args.coverage_num, args.pick_place_num,
                                                num_samples=sample_num, random_seed=12, overlap=True),
                              batch_size=2, shuffle=False, num_workers=1, pin_memory=True)
    # dataset = Mission.MissionDataset(num_visit=args.visiting_num, num_coverage=args.coverage_num, num_pick_place=args.pick_place_num, num_samples=sample_num)

    # Calculating heuristics of single sample
    heuristic_distance = torch.zeros(sample_num)
    solutions = []
    pointsets = []
    for i, tasksets in tqdm(train_loader):


        # TEST PART
        state = TA_State.initialize(tasksets.cuda())
        # print("current task")
        # print(state.cur_task)
        # print("given mission")
        # print(state.mission)
        # print("spent time")
        # print(state.spend_time)
        # print("previous task")
        # print(state.prev_task)
        # print("visited task")
        # print(state.visited)
        # print("step of the mission")
        # print(state.step)
        # print("time to depot of all task")
        # print(state.cost2depot)
        # print("GET MASK")
        # print(state.get_mask())
        # time_budget = time_budget_constraint(state.cur_task, state.mission)
        # print(time_budget)
        break

    print("Init===========")
    print("Mask")
    print(state.get_mask())
    print(state.get_remained_time())
    # selection = torch.LongTensor([1,1]).cuda()
    # state = state.update(selection)
    print(state.all_finished())
    # print("Step===========")
    # print("Mask")
    # print(state.get_mask())
    # print(state.spend_time)

