from Environment.State import TA_State
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from modules import Attention, GraphEmbedding


class RNNTSP(nn.Module):
    def __init__(self,
            embedding_size,
            hidden_size,
            seq_len,
            n_glimpses,
            tanh_exploration):
        super(RNNTSP, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_glimpses = n_glimpses
        self.seq_len = seq_len

        self.embedding = GraphEmbedding(8, embedding_size)
        self.encoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embedding_size+1, hidden_size, batch_first=True)
        self.pointer = Attention(hidden_size, C=tanh_exploration)
        self.glimpse = Attention(hidden_size)

        self.decoder_start_input = nn.Parameter(torch.FloatTensor(embedding_size))
        self.decoder_start_input.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))

    def apply_mask_to_logits(self, logits, mask, idxs):
        batch_size = logits.size(0)
        clone_mask = mask.clone()
        if idxs is not None:
            clone_mask[[i for i in range(batch_size)], idxs.data] = 1
            logits[clone_mask] = -np.inf
        return logits, clone_mask

    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size x seq_len x 8]
        """
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]

        embedded = self.embedding(inputs)
        encoder_outputs, (hidden, context) = self.encoder(embedded)

        prev_chosen_logprobs = []
        preb_chosen_indices = []
        prev_probs = []
        logits_matrix = []
        state = TA_State.initialize(inputs)
        preb_chosen_indices.append(state.prev_task.squeeze(1))


        decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1)

        while not state.all_finished():
            mask = state.get_mask()
            remained_time= state.get_remained_time()
            decoder_input = torch.cat((decoder_input, remained_time),dim=-1)
            _, (hidden, context) = self.decoder(decoder_input.unsqueeze(1), (hidden, context))

            query = hidden.squeeze(0)
            for _ in range(self.n_glimpses):
                ref, logits = self.glimpse(query, encoder_outputs)
                _mask = mask.clone()
                logits[_mask] = -100000.0
                query = torch.matmul(ref.transpose(-1, -2), torch.softmax(logits, dim=-1).unsqueeze(-1)).squeeze(-1)

            _, logits = self.pointer(query, encoder_outputs)

            _mask = mask.clone()
            logits[_mask] = -100000.0
            probs = torch.softmax(logits, dim=-1)
            cat = Categorical(probs)
            chosen = cat.sample()
            mask[[i for i in range(batch_size)], chosen] = True
            log_probs = cat.log_prob(chosen)
            decoder_input = embedded.gather(1, chosen[:, None, None].repeat(1, 1, self.hidden_size)).squeeze(1)
            prev_chosen_logprobs.append(log_probs)
            preb_chosen_indices.append(chosen)
            prev_probs.append(probs)
            logits_matrix.append(logits)

            state = state.update(chosen)

        return torch.stack(prev_probs, 1),  torch.stack(prev_chosen_logprobs, 1),  torch.stack(preb_chosen_indices, 1), torch.stack(logits_matrix, 1)
