"""
policy_net.py — REINFORCE용 정책 신경망

Actor만 있고 Critic이 없다. Monte Carlo return G_t를 직접 사용.
cartpole/pg_agent.py의 PolicyNet 기반.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class PolicyNet(nn.Module):

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=(2 ** 0.5))
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> Categorical:
        logits = self.net(obs)
        return Categorical(logits=logits)
