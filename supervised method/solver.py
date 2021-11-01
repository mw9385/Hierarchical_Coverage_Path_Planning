import math

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rl_with_attention import AttentionTSP
from beta_rl_with_attention import Beta_AttentionTSP
from rl_with_rnn import RNNTSP
from rl_mission_planning import HetNet

from Environment.utils import get_solution_reward


class Solver(nn.Module):
    def __init__(self):
        super(Solver, self).__init__()

    def reward(self, sample_solution):
        """
        [Input]
            sample_solution: batch x seq_len x feature_size -> torch.LongTensor
        [Return]
            tour_len: batch
        """

        tour_len = get_solution_reward(sample_solution)

        return tour_len

    def forward(self, inputs):
        """
        [Input]
            inputs: batch x seq_len x feature
        [Return]
            R:
            probs:
            actions: batch x seq_len
        """

        raw_probs, probs, actions, logits  = self.actor(inputs)

        R = self.reward(inputs.gather(1, actions.unsqueeze(2).repeat(1, 1, inputs.shape[-1])))

        return raw_probs, probs, actions, R, logits


class solver_RNN(Solver):
    def __init__(self,
            embedding_size,
            hidden_size,
            seq_len,
            n_glimpses,
            tanh_exploration):
        super(solver_RNN, self).__init__()

        self.actor = RNNTSP(embedding_size,
                                hidden_size,
                                seq_len,
                                n_glimpses,
                                tanh_exploration)

class solver_Attention(Solver):
    def __init__(self,
            embedding_size,
            hidden_size,
            seq_len,
            n_glimpses,
            tanh_exploration):
        super(solver_Attention, self).__init__()

        self.actor = AttentionTSP(embedding_size,
                                  hidden_size,
                                  seq_len)


class Beta_solver_Attention(Solver):
    def __init__(self,
            embedding_size,
            hidden_size,
            seq_len,
            n_glimpses,
            tanh_exploration):
        super(Beta_solver_Attention, self).__init__()

        self.actor = Beta_AttentionTSP(embedding_size,
                                  hidden_size,
                                  seq_len)

class HetNet_solver(Solver):
    def __init__(self,
                embedding_size,
                hidden_size,
                n_glimpses,
                tanh_exploration):
            super(HetNet_solver, self).__init__()

            self.actor = HetNet(embedding_size,
                                    hidden_size)
