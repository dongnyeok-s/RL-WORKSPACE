"""
q_network.py — DQN용 Q-Network

Q(s, a) 를 추정하는 MLP.
Double DQN에서는 동일 구조의 online / target 네트워크 2개를 사용한다.
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs → Q(s, ·)  shape: (batch, action_dim)"""
        return self.net(obs)
