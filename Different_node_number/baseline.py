import torch
import torch.nn as nn
import torch.nn.functional as F
from Attention_model import Encoder

class Baseline(object):

    def wrap_dataset(self, dataset):
        return dataset

    def unwrap_batch(self, batch):
        return batch, None

    def eval(self, x, c):
        raise NotImplementedError("Override this method")

    def get_learnable_parameters(self):
        return []

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

class CriticBaseline(Baseline):
    def __init__(self, critic):
        super(Baseline, self).__init__()
        self.critic = critic

    def eval(self, x, c):
        v = self.critic(x)
        return v.detach(), F.mse_loss(v, c.detach())
    
    def get_learnable_parameters(self):
        return list(self.critic.parameters())

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {
            'critic':self.critic.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        critic_state_dict = state_dict.get('critic', {})
        if not isinstance(critic_state_dict, dict):  # backwards compatibility
            critic_state_dict = critic_state_dict.state_dict()
        self.critic.load_state_dict({**self.critic.state_dict(), **critic_state_dict})

class CriticNetwork(nn.Module):
    def __init__(
        self,
        n_hidden,        
    ):

        super(CriticNetwork, self).__init__()
        self.n_hidden = n_hidden        
        
        self.encoder = Encoder(n_feature=2, n_hidden=n_hidden, high_level=False)
        self.value_head = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, 1)
        )
    def forward(self, inputs):
        low_value = []
        low_embed, high_embed, _ = self.encoder(inputs)
        high_value = self.value_head(high_embed).squeeze(2)
        
        for cell_samples in low_embed:
            for node_samples in cell_samples:
                _low_value = self.value_head(node_samples)
                print(_low_value.squeeze(2))
                low_value.append(_low_value.squeeze(2))
        low_value = torch.stack(low_value, 0)
        return self.value_head(high_embed)
