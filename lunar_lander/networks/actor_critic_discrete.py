"""
actor_critic_discrete.py — 이산 행동 Actor-Critic (PPO용)

cartpole/network.py 기반, LunarLander-v3에 맞게 조정.
  - obs_dim    = 8  (x, y, vx, vy, angle, angular_vel, left_leg, right_leg)
  - action_dim = 4  (noop, left engine, main engine, right engine)

공유 Backbone + Actor Head (Categorical) + Critic Head (V(s))
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


def _init_weights(layer: nn.Linear, gain: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCriticDiscrete(nn.Module):

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.backbone = nn.Sequential(
            _init_weights(nn.Linear(obs_dim, hidden_dim), gain=(2 ** 0.5)),
            nn.Tanh(),
            _init_weights(nn.Linear(hidden_dim, hidden_dim), gain=(2 ** 0.5)),
            nn.Tanh(),
        )
        self.actor_head = _init_weights(nn.Linear(hidden_dim, action_dim), gain=0.01)
        self.critic_head = _init_weights(nn.Linear(hidden_dim, 1), gain=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        features = self.backbone(obs)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        dist = Categorical(logits=logits)
        return dist, value

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor) -> tuple[int, float, float]:
        dist, value = self.forward(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.squeeze().item()

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.forward(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, value.squeeze(-1), entropy
