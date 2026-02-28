"""
reinforce_agent.py — REINFORCE (Vanilla Policy Gradient) 에이전트

에피소드 단위로 Monte Carlo return G_t를 계산한 뒤 정책 업데이트.
Critic 없이 Actor만 사용.
"""

import torch
from torch.optim import Adam

from ..networks.policy_net import PolicyNet


class REINFORCEAgent:

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        device: torch.device = torch.device("cpu"),
    ):
        self.gamma = gamma
        self.device = device

        self.policy = PolicyNet(obs_dim, action_dim, hidden_dim).to(device)
        self.optimizer = Adam(self.policy.parameters(), lr=lr)

        self._log_probs: list[torch.Tensor] = []
        self._rewards: list[float] = []

    def select_action(self, obs_np) -> int:
        obs_tensor = torch.tensor(obs_np, dtype=torch.float32).to(self.device)
        dist = self.policy(obs_tensor)
        action = dist.sample()
        self._log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward: float) -> None:
        self._rewards.append(reward)

    def update(self) -> dict:
        T = len(self._rewards)

        # G_t 역방향 계산
        returns = torch.zeros(T, device=self.device)
        G = 0.0
        for t in reversed(range(T)):
            G = self._rewards[t] + self.gamma * G
            returns[t] = G

        # 정규화
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs = torch.stack(self._log_probs)
        loss = -(log_probs * returns).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        ep_reward = sum(self._rewards)
        self._log_probs = []
        self._rewards = []

        return {"loss": loss.item(), "ep_reward": ep_reward, "ep_length": T}

    def save(self, path: str) -> None:
        torch.save(self.policy.state_dict(), path)
        print(f"모델 저장: {path}")

    def load(self, path: str) -> None:
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.policy.eval()
        print(f"모델 로드: {path}")
