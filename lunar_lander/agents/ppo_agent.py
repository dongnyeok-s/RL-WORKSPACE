"""
ppo_agent.py — PPO 에이전트 (이산/연속 모드 지원)

이산 모드 (LunarLander-v3):
  - ActorCriticDiscrete (Categorical 분포)

연속 모드 (LunarLanderContinuous-v3):
  - ActorCriticContinuous (Diagonal Gaussian 분포)
  → Phase 3에서 추가
"""

import torch
import torch.nn as nn
from torch.optim import Adam

from ..buffers.rollout_buffer import RolloutBuffer


class PPOAgent:

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        continuous: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.continuous = continuous
        self.device = device

        if continuous:
            from ..networks.actor_critic_continuous import ActorCriticContinuous
            self.network = ActorCriticContinuous(obs_dim, action_dim, hidden_dim).to(device)
        else:
            from ..networks.actor_critic_discrete import ActorCriticDiscrete
            self.network = ActorCriticDiscrete(obs_dim, action_dim, hidden_dim).to(device)

        self.optimizer = Adam(self.network.parameters(), lr=lr)

    def select_action(self, obs_np):
        obs_tensor = torch.tensor(obs_np, dtype=torch.float32).to(self.device)
        return self.network.get_action(obs_tensor)

    @torch.no_grad()
    def get_value(self, obs_np) -> float:
        obs_tensor = torch.tensor(obs_np, dtype=torch.float32).to(self.device)
        _, value = self.network(obs_tensor)
        return value.squeeze().item()

    def update(self, buffer: RolloutBuffer, n_epochs: int, batch_size: int) -> dict:
        self.network.train()

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        n_updates = 0

        for _ in range(n_epochs):
            for batch in buffer.get_minibatches(batch_size):
                obs = batch["obs"]
                actions = batch["actions"]
                old_log_probs = batch["log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                new_log_probs, values, entropy = self.network.evaluate_actions(obs, actions)

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.functional.mse_loss(values, returns)
                entropy_loss = -entropy.mean()

                loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_log_probs - new_log_probs).mean().item()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += (-entropy_loss).item()
                total_approx_kl += approx_kl
                n_updates += 1

        self.network.eval()

        return {
            "actor_loss": total_actor_loss / n_updates,
            "critic_loss": total_critic_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "approx_kl": total_approx_kl / n_updates,
        }

    def save(self, path: str) -> None:
        torch.save(self.network.state_dict(), path)
        print(f"모델 저장: {path}")

    def load(self, path: str) -> None:
        self.network.load_state_dict(torch.load(path, map_location=self.device))
        self.network.eval()
        print(f"모델 로드: {path}")
