"""
dqn_agent.py — Double DQN 에이전트

핵심:
  - Online Q-net: action 선택 (argmax Q_online)
  - Target Q-net: value 평가 (Q_target[s', a*])
  - ε-greedy 탐험: ε를 1.0 → 0.05로 선형 감소 (50K steps)
  - Polyak soft target update: θ_target ← τ·θ_online + (1-τ)·θ_target

Double DQN Loss:
  a* = argmax_a Q_online(s', a)
  y  = r + γ · Q_target(s', a*) · (1 - done)
  L  = MSE(Q_online(s, a), y)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from ..networks.q_network import QNetwork
from ..buffers.replay_buffer import ReplayBuffer


class DQNAgent:

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: int = 50_000,
        device: torch.device = torch.device("cpu"),
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.device = device

        # Online / Target 네트워크
        self.q_online = QNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.q_target = QNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()

        self.optimizer = Adam(self.q_online.parameters(), lr=lr)

        self._step = 0

    def _epsilon(self) -> float:
        """선형 ε 감소 스케줄."""
        fraction = min(1.0, self._step / self.eps_decay)
        return self.eps_start + fraction * (self.eps_end - self.eps_start)

    def select_action(self, obs_np) -> int:
        """ε-greedy 행동 선택."""
        self._step += 1
        eps = self._epsilon()
        if np.random.random() < eps:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            obs_tensor = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.q_online(obs_tensor)
            return q_values.argmax(dim=1).item()

    @torch.no_grad()
    def greedy_action(self, obs_np) -> int:
        """평가용 greedy 행동 선택 (ε=0)."""
        obs_tensor = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        q_values = self.q_online(obs_tensor)
        return q_values.argmax(dim=1).item()

    def update(self, buffer: ReplayBuffer, batch_size: int) -> dict:
        """Double DQN 업데이트 1회."""
        batch = buffer.sample(batch_size)
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        # Online Q(s, a)
        q_values = self.q_online(obs)
        q_a = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            # Online net → action 선택
            next_actions = self.q_online(next_obs).argmax(dim=1)
            # Target net → value 평가
            next_q = self.q_target(next_obs).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards + self.gamma * next_q * (1.0 - dones)

        loss = nn.functional.mse_loss(q_a, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_online.parameters(), 10.0)
        self.optimizer.step()

        # Polyak soft target update
        self._soft_update()

        return {"loss": loss.item(), "epsilon": self._epsilon(), "q_mean": q_a.mean().item()}

    def _soft_update(self):
        for p_online, p_target in zip(self.q_online.parameters(), self.q_target.parameters()):
            p_target.data.mul_(1.0 - self.tau).add_(p_online.data * self.tau)

    def save(self, path: str) -> None:
        torch.save(self.q_online.state_dict(), path)
        print(f"모델 저장: {path}")

    def load(self, path: str) -> None:
        self.q_online.load_state_dict(torch.load(path, map_location=self.device))
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_online.eval()
        self.q_target.eval()
        print(f"모델 로드: {path}")
