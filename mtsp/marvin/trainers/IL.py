'''
Copyright (c) 2020 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project.

See the License for the specific language governing permissions and
limitations under the License.
'''

'''
Here we have the imitation learning trainer. Here, the only function that is really
different is the function that performs the batch step.
'''

import math
from random import seed

import numpy as np
import torch

from marvin.graph_gen.generate import DataGenerator
from marvin.models.general import device
from marvin.traffic_simulation.traffic_sim import perform_timestep
from marvin.trainers.generic import Trainer
from marvin.utils.utils import optimized

seed(0)


class ILTrainer(Trainer):
    def initialize_constants(self):
        """Adds the trainer name for the tensorboard file"""

        super().initialize_constants()
        self.trainer = "IL"

    def next_epoch(self):
        """Not needed for IL"""
        pass

    def run_batch(self):
        """Run and calculate the loss for one batch

        Returns:
            Tuple[torch.Tensor, float, str] --
                1: the loss of the batch to be minized
                2: The accuracy of the agent's actions
                3: name of the above metric
        """

        # allow the model to run in train mode
        self.model = self.model.train()
        accuracies = []
        il_loss = torch.tensor([], device=device)

        for _ in range(self.args.batch_size):
            if self.args.random_dataset: # default는 true
                graph = self.allocate_graph(DataGenerator.random_dataset(1, self.args)[0])
            else:
                graph = self.allocate_graph(DataGenerator.get_graph(self.args))

            self.args.num_nodes = graph.adj.shape[0]            
            self.reset_models()
            # action masking 작업 / agent의 갯수 만큼 node 생성
            undiscovered = [torch.ones(
                self.args.num_nodes, device=device).float()
                            for _ in range(self.args.num_agents)]

            # how many times each node must be traversed        
            pass_count = torch.tensor([1 for _ in range(self.args.num_nodes)],
                                        device=device).float()                                                                  
            congestion = np.random.uniform(0, 1, self.args.num_nodes)
            capacities = np.ones(self.args.num_nodes)
            simple_adj = graph.adj.cpu().numpy()

            in_routes = (graph.adj.cpu().numpy() > 0).sum(0)

            # Normalize the cost to the orginal street lengths
            solution_matrix = graph.adj.cpu().numpy() * \
                float(graph.actual_distance.max() / graph.dense.max())          
            if self.args.traffic:
                # calculate what a realistic set of congestion states would be
                for _ in range(10):
                    congestion, edge_val = perform_timestep(
                        simple_adj, capacities, congestion, in_routes
                    )

                solution_matrix = (solution_matrix.transpose() /
                                   np.clip(1 - congestion ** 3, 0.25, 1)
                                   ).transpose()

            # calculate the optimal cost of this traversal            
            opt_cost, opt_path = optimized.LKH(
                solution_matrix, pass_count=pass_count,
                num_agents=self.args.num_agents,
                identifier=self.identifier, is_tour=True)
            pos = []
            for s in range(len(opt_path)):
                pos.append(opt_path[s][0])
                opt_path[s] = opt_path[s][1:]

            # Initialize the location of the agents as being discovered
            for i in range(self.args.num_agents):
                undiscovered[i][pos[i]] = 0.0
                pass_count[pos[i]] = 0.0

            static_features = self.get_static_features(graph)

            correct_steps = 0
            steps = 0
            
            costs_so_far = [0 for _ in range(self.args.num_agents)] # 0을 agent 갯수 만큼 채워줌 
            # opt_path는 agent 갯수 만큼 route가 나오는데, route 마다 길이가 다르기 때문에 가장 긴 route에 맞춰서 그만큼 iteration을 돌림
            while pass_count.sum() > 0 and \
                    max(map(lambda x: len(x), opt_path)) > 0:
                
                # agent 중 어느 agent에게 일을 할당할지 결정함
                # agent 중 거리가 더 짧은 agent를 선택해서 prediction을 진행함
                # 둘다 0인 상황에서는 1번 agent가 먼저 시작함
                i = costs_so_far.index(min(costs_so_far))

                if pass_count.sum() <= 0 or len(opt_path[i]) == 0:
                    costs_so_far[i] += math.inf
                    continue

                dynamic_features = self.get_dynamic_features(graph, pos[i],
                    undiscovered[i], pass_count, congestion)

                feature = torch.cat((static_features, dynamic_features), 1)

                # normalize the features
                feature = feature / (torch.clamp(
                    feature.max(0)[0], min=1.)
                )

                mask = (pass_count > 0).float()
                prediction = self.model(
                    feature, graph, i, pos[i], mask=mask
                )
                # we force the next step to be the actual next step
                # in the solution            
                next_step = opt_path[i][0]
                opt_path[i] = opt_path[i][1:]                

                # for accuracy calculation
                steps += 1
                correct_steps += \
                    float(torch.argmax(prediction)) == next_step

                # add the imination learning loss to the set
                if il_loss.shape != (0,):
                    il_pred = prediction.unsqueeze(0)                    
                    il_sol = torch.tensor(
                        next_step, device=device).unsqueeze(0)
                    il_loss = torch.cat((
                        il_loss,
                        self.ce(il_pred, il_sol
                                ).unsqueeze(0)), 0)
                else:
                    il_pred = prediction.unsqueeze(0)
                    il_sol = torch.tensor(
                        next_step, device=device).unsqueeze(0)
                    il_loss = self.ce(
                        il_pred, il_sol
                        ).unsqueeze(0)

                action = int(next_step)

                if self.args.asynch:
                    costs_so_far[i] += solution_matrix[pos[i], action]
                else:
                    costs_so_far[i] += 1

                # update the agent's actions
                pos[i] = action
                undiscovered[i][action] = 0.0
                # pass_count[action] = max(0, pass_count[action] - 1)

            accuracies.append(correct_steps / steps)
        loss = il_loss.mean()
        return loss, sum(accuracies) / len(accuracies), "Mean Accuracy"
