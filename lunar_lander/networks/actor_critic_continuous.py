"""
actor_critic_continuous.py — 연속 행동 Actor-Critic (PPO-Continuous용)

Diagonal Gaussian 정책:
  - mean = actor_head(features)
  - log_std = 학습 가능한 파라미터 (state-independent)
  - action ~ Normal(mean, exp(log_std))

LunarLanderContinuous-v3:
  - obs_dim    = 8
  - action_dim = 2  (main engine, lateral engine)
  - action range: [-1, 1]
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


def _init_weights(layer: nn.Linear, gain: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCriticContinuous(nn.Module):

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.action_dim = action_dim

        self.backbone = nn.Sequential(
            _init_weights(nn.Linear(obs_dim, hidden_dim), gain=(2 ** 0.5)),
            nn.Tanh(),
            _init_weights(nn.Linear(hidden_dim, hidden_dim), gain=(2 ** 0.5)),
            nn.Tanh(),
        )
        self.actor_mean = _init_weights(nn.Linear(hidden_dim, action_dim), gain=0.01)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic_head = _init_weights(nn.Linear(hidden_dim, 1), gain=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        features = self.backbone(obs)
        mean = self.actor_mean(features)
        std = self.actor_log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        value = self.critic_head(features)
        return dist, value

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor) -> tuple:
        """
        Returns:
            action   : numpy array (action_dim,)
            log_prob : float (합산된 log_prob)
            value    : float
        """
        dist, value = self.forward(obs)
        action = dist.sample()
        # 독립 차원의 log_prob 합산
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.cpu().numpy(), log_prob.item(), value.squeeze().item()

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.forward(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, value.squeeze(-1), entropy
