"""
sac_networks.py — 드론 투하용 SAC 네트워크

lunar_lander/networks/sac_networks.py 기반.
입출력 차원: obs=14, action=4.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class TwinQNetwork(nn.Module):
    """독립적인 Q-network 2개."""

    def __init__(self, obs_dim: int = 14, action_dim: int = 4, hidden_dim: int = 256):
        super().__init__()
        input_dim = obs_dim + action_dim

        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa), self.q2(sa)


class GaussianPolicy(nn.Module):
    """Squashed Gaussian Policy for SAC."""

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, obs_dim: int = 14, action_dim: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor):
        features = self.net(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()

        dist = Normal(mean, std)
        u = dist.rsample()
        action = torch.tanh(u)

        log_prob = dist.log_prob(u).sum(dim=-1, keepdim=True)
        log_prob -= (2 * (torch.log(torch.tensor(2.0)) - u - nn.functional.softplus(-2 * u))).sum(dim=-1, keepdim=True)

        mean_action = torch.tanh(mean)

        return action, log_prob, mean_action
