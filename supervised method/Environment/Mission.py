import torch
import pickle
import numpy as np
import random

from typing import List
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class MissionDataset(Dataset):

    def __init__(self, num_visit = 0, num_coverage = 0, num_pick_place = 0, num_samples = 0, random_seed=111, overlap=True):
        super(MissionDataset, self).__init__()

        assert num_visit > 0 or num_coverage > 0 or num_pick_place > 0, "at least one task"
        assert num_samples > 0, "at least one samples"

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        self.data_set = []

        for l in tqdm(range(num_samples)):
            task_list = []

            # Generate area coverage task
            if num_coverage > 0:
                area_list = []
                while len(area_list) < num_coverage:
                    point = torch.FloatTensor(np.random.uniform(0.1, 0.9, 2))
                    area_info = torch.FloatTensor(np.random.uniform(0.05, 0.08, 1))

                    # # Non-overlapping
                    if overlap is False:
                        if any(np.linalg.norm(point-A[:2]) < area_info + A[2] + 0.05 for A in area_list):
                            continue

                    data = torch.cat((point, area_info, point.clone(), torch.FloatTensor([0,0,1])), axis=0)

                    area_list.append(data)

                task_list = task_list + area_list

            # Generate point visitation task
            if num_visit > 0:
                visit_list = []
                while len(visit_list) < num_visit:
                    point = torch.FloatTensor(np.random.uniform(0.1, 0.99, 2))
                    area_info = torch.zeros(1)

                    # Non-overlapping
                    if overlap is False:
                        if num_coverage > 0:
                            if any(np.linalg.norm(point - A[:2]) < A[2] + 0.01 for A in area_list):
                                continue

                    data = torch.cat((point, area_info, point.clone(), torch.FloatTensor([0,1,0])), axis=0)

                    visit_list.append(data)

                task_list = task_list + visit_list

            # Generate Pick&Place task
            if num_pick_place > 0:
                pick_place_list = []
                while len(pick_place_list) < num_pick_place:
                    pick_point = torch.FloatTensor(np.random.uniform(0.1,0.99,2))
                    place_point = pick_point + ([(-1)**random.randint(0,1), (-1)**random.randint(0,1)]*np.random.uniform(0.02, 0.09, 2)).astype(np.float32)
                    area_info = torch.FloatTensor([np.linalg.norm(pick_point-place_point)])

                    # Non-overlapping
                    if overlap is False:
                        if num_coverage > 0:
                            if any((np.linalg.norm(pick_point - A[:2]) < A[2] + 0.01) or (np.linalg.norm(place_point - A[:2]) < A[2] + 0.01) for A in area_list):
                                continue

                    data = torch.cat((pick_point, area_info, place_point, torch.FloatTensor([1,0,0])), axis=0)

                    pick_place_list.append(data)

                task_list = task_list + pick_place_list

            task_list = torch.stack(task_list)

            self.data_set.append(torch.cat((torch.zeros(1, 8), task_list),axis=0))

        self.size = len(self.data_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return idx, self.data_set[idx]


class MissionDataset_General(Dataset):

    def __init__(self, task_num_list=None, batch_size=None, chunk_size=None,
                       random_seed=111, overlap=True):
        super(MissionDataset_General, self).__init__()

        assert isinstance(task_num_list, list)
        assert len(task_num_list) > 0, 'task_num_list should be list'
        assert batch_size > 0, "at least one batch_size"
        assert chunk_size > 0, "at least one chunk"


        # Fix random seed
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        np.random.seed(random_seed)
        random.seed(random_seed)

        num_sample = batch_size*chunk_size*len(task_num_list)
        print("---sample num: %d"%(num_sample))


        self.data_set = []

        for task_num in task_num_list:

            num_visit, num_coverage, num_pick_place = task_num, task_num, task_num

            print("==Task num: %d=="%(task_num))
            for chunck in tqdm(range(chunk_size)):
                chunk_list = []
                for b in range(batch_size):
                    task_list = []
                    # Generate area coverage task
                    if num_coverage > 0:
                        area_list = []
                        while len(area_list) < num_coverage:
                            point = torch.FloatTensor(np.random.uniform(0.1, 0.9, 2))
                            area_info = torch.FloatTensor(np.random.uniform(0.05, 0.08, 1))

                            # # Non-overlapping
                            if overlap is False:
                                if any(np.linalg.norm(point-A[:2]) < area_info + A[2] + 0.05 for A in area_list):
                                    continue

                            data = torch.cat((point, area_info, point.clone(), torch.FloatTensor([0,0,1])), axis=0)

                            area_list.append(data)

                        task_list = task_list + area_list

                    # Generate point visitation task
                    if num_visit > 0:
                        visit_list = []
                        while len(visit_list) < num_visit:
                            point = torch.FloatTensor(np.random.uniform(0.1, 0.99, 2))
                            area_info = torch.zeros(1)

                            # Non-overlapping
                            if overlap is False:
                                if num_coverage > 0:
                                    if any(np.linalg.norm(point - A[:2]) < A[2] + 0.01 for A in area_list):
                                        continue

                            data = torch.cat((point, area_info, point.clone(), torch.FloatTensor([0,1,0])), axis=0)

                            visit_list.append(data)

                        task_list = task_list + visit_list

                    # Generate Pick&Place task
                    if num_pick_place > 0:
                        pick_place_list = []
                        while len(pick_place_list) < num_pick_place:
                            pick_point = torch.FloatTensor(np.random.uniform(0.1,0.99,2))
                            place_point = pick_point + ([(-1)**random.randint(0,1), (-1)**random.randint(0,1)]*np.random.uniform(0.02, 0.09, 2)).astype(np.float32)
                            area_info = torch.FloatTensor([np.linalg.norm(pick_point-place_point)])

                            # Non-overlapping
                            if overlap is False:
                                if num_coverage > 0:
                                    if any((np.linalg.norm(pick_point - A[:2]) < A[2] + 0.01) or (np.linalg.norm(place_point - A[:2]) < A[2] + 0.01) for A in area_list):
                                        continue

                            data = torch.cat((pick_point, area_info, place_point, torch.FloatTensor([1,0,0])), axis=0)

                            pick_place_list.append(data)

                        task_list = task_list + pick_place_list

                    task_list = torch.stack(task_list) # (num_task*3) X 8

                    chunk_list.append(torch.cat((torch.zeros(1, 8), task_list),axis=0))

                self.data_set.append(torch.stack(chunk_list))


        self.size = len(self.data_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return idx, self.data_set[idx]




def test(args):
    train_loader = DataLoader(MissionDataset(args.visiting_num, args.coverage_num, args.pick_place_num,
                                                num_samples=1, random_seed=12, overlap=True),
                              batch_size=1, shuffle=False, num_workers=1)
    for  (batch_idx, task_list_batch) in train_loader:
        print(task_list_batch)
        print(task_list_batch.shape)

def test_general(args):
    train_loader = DataLoader(MissionDataset_General([1,3,5], args.batch_size, args.chunk_size),
                              batch_size=1, shuffle=False, num_workers=1)

    for (batch_idx, batch_data) in train_loader:
        print(batch_data.shape)
        # print(batch_data)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("--visiting_num", type=int, default=2)
    # parser.add_argument("--coverage_num", type=int, default=2)
    # parser.add_argument("--pick_place_num", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--chunk_size", type=int, default=2)
    args = parser.parse_args()

    test_general(args)


class PointDataset(Dataset):
    def __init__(self, train_x, train_y):
        super(PointDataset, self).__init__()
        self.train_x = train_x
        self.train_y = train_y        
        self.train_x = self.train_x.transpose(1,0)
        self.train_y = self.train_y.transpose(0,1)
        self.size = len(self.train_y)
                
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        self.x = self.train_x[idx]
        self.y = self.train_y[idx]   
        return self.x, self.y    