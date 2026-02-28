"""
sac_networks.py — SAC용 Twin Q-Network + Squashed Gaussian Policy

Twin Q:
  Q1(s, a), Q2(s, a) → min(Q1, Q2) 로 과대추정 방지

Squashed Gaussian Policy:
  u ~ Normal(mean, std)
  a = tanh(u)  → [-1, 1]
  log_prob 보정: log π(a|s) = log π(u|s) - sum( log(1 - tanh²(u) + ε) )
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class TwinQNetwork(nn.Module):
    """독립적인 Q-network 2개."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
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
        """Returns: (q1_value, q2_value)  each shape (batch, 1)"""
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa), self.q2(sa)


class GaussianPolicy(nn.Module):
    """Squashed Gaussian Policy for SAC."""

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor):
        """
        Returns:
            action      : (batch, action_dim) tanh-squashed action
            log_prob    : (batch, 1) corrected log probability
            mean_action : (batch, action_dim) tanh(mean) — 평가용 deterministic action
        """
        features = self.net(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()

        dist = Normal(mean, std)
        # Reparameterization trick
        u = dist.rsample()
        action = torch.tanh(u)

        # Log-prob with tanh correction
        log_prob = dist.log_prob(u).sum(dim=-1, keepdim=True)
        log_prob -= (2 * (torch.log(torch.tensor(2.0)) - u - nn.functional.softplus(-2 * u))).sum(dim=-1, keepdim=True)

        mean_action = torch.tanh(mean)

        return action, log_prob, mean_action
