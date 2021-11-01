from math import sqrt

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset

from modules import Glimpse, GraphEmbedding, Pointer

from Environment.State import TA_State


class att_layer(nn.Module):
    def __init__(self, embed_dim, n_heads, feed_forward_hidden=512, bn=False):
        super(att_layer, self).__init__()
        self.mha = torch.nn.MultiheadAttention(embed_dim, n_heads)
        self.embed = nn.Sequential(nn.Linear(embed_dim, feed_forward_hidden), nn.ReLU(), nn.Linear(feed_forward_hidden, embed_dim))

    def forward(self, x):
        """
        [Input]
            x: batch X seq_len X embedding_size
        [Output]
            _2: batch X seq_len X embedding_size

            # Multiheadattention in pytorch starts with (target_seq_length, batch_size, embedding_size).
            # thus we permute order first. https://pytorch.org/docs/stable/nn.html#multiheadattention
        """
        x = x.permute(1, 0, 2)
        _1 = x + self.mha(x, x, x)[0]
        _1 = _1.permute(1, 0, 2)
        _2 = _1 + self.embed(_1)
        return _2


class AttentionModule(nn.Sequential):
    def __init__(self, embed_dim, n_heads, feed_forward_hidden=512, n_self_attentions=3, bn=False):
        super(AttentionModule, self).__init__(
            *(att_layer(embed_dim, n_heads, feed_forward_hidden, bn) for _ in range(n_self_attentions))
        )


class AttentionTSP(nn.Module):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 seq_len,
                 n_head=8,
                 C=10):
        super(AttentionTSP, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        # self.seq_len = seq_len
        self.n_head = n_head
        self.C = C

        self.embedding = GraphEmbedding(8, embedding_size)
        self.mha = AttentionModule(embedding_size, n_head)

        self.init_w = nn.Parameter(torch.Tensor(2 * self.embedding_size))
        self.init_w.data.uniform_(-1, 1)
        self.glimpse = Glimpse(self.embedding_size, self.hidden_size, self.n_head)
        self.pointer = Pointer(self.embedding_size, self.hidden_size, 1, self.C)

        self.h_context_embed = nn.Linear(self.embedding_size, self.embedding_size)
        self.v_weight_embed = nn.Linear(self.embedding_size * 2, self.embedding_size)

    def forward(self, inputs):
        """
        [Input]
            inputs: batch_size x seq_len x feature_size
        [Return]
            logprobs:
            actions:
        """
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]

        embedded = self.embedding(inputs) # batch x task_num x embedding_size
        h = self.mha(embedded) # batch x task_num x embedding_size
        h_mean = h.mean(dim=1) # batch x embedding_size
        h_bar = self.h_context_embed(h_mean)
        h_rest = self.v_weight_embed(self.init_w)
        query = h_bar + h_rest

        #init query
        prev_chosen_indices = []
        prev_chosen_logprobs = []
        first_chosen_hs = None

        state = TA_State.initialize(inputs)

        # mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        # mask[:,0] = True # Except Depot at first time

        # depot = torch.zeros(batch_size,dtype=torch.int64 ,device=inputs.device)
        prev_chosen_indices.append(state.prev_task.squeeze(1))

        while not state.all_finished():
            mask = state.get_mask() # batch x task_num

            _, n_query = self.glimpse(query, h, mask)
            prob, _ = self.pointer(n_query, h, mask)
            cat = Categorical(prob)
            chosen = cat.sample() # batch
            logprobs = cat.log_prob(chosen)
            prev_chosen_indices.append(chosen)
            prev_chosen_logprobs.append(logprobs)

            # mask[[i for i in range(batch_size)], chosen] = True
            state = state.update(chosen)

            cc = chosen.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.embedding_size)
            if first_chosen_hs is None:
                first_chosen_hs = h.gather(1, cc).squeeze(1)
            chosen_hs = h.gather(1, cc).squeeze(1)
            h_rest = self.v_weight_embed(torch.cat([first_chosen_hs, chosen_hs], dim=-1))
            query = h_bar + h_rest

        return torch.stack(prev_chosen_logprobs, 1), torch.stack(prev_chosen_indices, 1)
