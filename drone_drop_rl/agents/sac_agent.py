"""
sac_agent.py — 드론 투하용 SAC 에이전트

lunar_lander/agents/sac_agent.py 기반.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from ..networks.sac_networks import TwinQNetwork, GaussianPolicy


class ReplayBuffer:
    """Off-Policy Replay Buffer (lunar_lander/buffers/replay_buffer.py 기반)."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int,
                 device: torch.device = torch.device("cpu")):
        self.capacity = capacity
        self.device = device
        self._ptr = 0
        self._size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def store(self, obs, action, reward, next_obs, done):
        idx = self._ptr % self.capacity
        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_obs[idx] = next_obs
        self.dones[idx] = float(done)
        self._ptr += 1
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        indices = np.random.randint(0, self._size, size=batch_size)
        return {
            "obs": torch.tensor(self.obs[indices], dtype=torch.float32).to(self.device),
            "actions": torch.tensor(self.actions[indices], dtype=torch.float32).to(self.device),
            "rewards": torch.tensor(self.rewards[indices], dtype=torch.float32).to(self.device),
            "next_obs": torch.tensor(self.next_obs[indices], dtype=torch.float32).to(self.device),
            "dones": torch.tensor(self.dones[indices], dtype=torch.float32).to(self.device),
        }

    def __len__(self):
        return self._size


class SACAgent:
    """SAC (Soft Actor-Critic) 에이전트."""

    def __init__(
        self,
        obs_dim: int = 14,
        action_dim: int = 4,
        hidden_dim: int = 256,
        lr_actor: float = 1e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        init_alpha: float = 0.2,
        target_entropy: float = -4.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy
        self.device = device

        self.policy = GaussianPolicy(obs_dim, action_dim, hidden_dim).to(device)
        self.q_nets = TwinQNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.q_target = TwinQNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.q_target.load_state_dict(self.q_nets.state_dict())
        self.q_target.eval()

        self.actor_optimizer = Adam(self.policy.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.q_nets.parameters(), lr=lr_critic)

        self.log_alpha = torch.tensor(
            np.log(init_alpha), dtype=torch.float32,
            device=device, requires_grad=True,
        )
        self.alpha_optimizer = Adam([self.log_alpha], lr=lr_alpha)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, obs_np) -> np.ndarray:
        """학습용: stochastic action."""
        with torch.no_grad():
            obs_t = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0).to(self.device)
            action, _, _ = self.policy(obs_t)
            return action.squeeze(0).cpu().numpy()

    def deterministic_action(self, obs_np) -> np.ndarray:
        """평가용: deterministic action (mean)."""
        with torch.no_grad():
            obs_t = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0).to(self.device)
            _, _, mean_action = self.policy(obs_t)
            return mean_action.squeeze(0).cpu().numpy()

    def update(self, buffer: ReplayBuffer, batch_size: int) -> dict:
        batch = buffer.sample(batch_size)
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"].unsqueeze(1)
        next_obs = batch["next_obs"]
        dones = batch["dones"].unsqueeze(1)

        alpha = self.alpha.detach()

        # ── Critic 업데이트 ──────────────────────────────────────────────
        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy(next_obs)
            q1_target, q2_target = self.q_target(next_obs, next_action)
            q_target_min = torch.min(q1_target, q2_target)
            target_q = rewards + self.gamma * (1.0 - dones) * (q_target_min - alpha * next_log_prob)

        q1, q2 = self.q_nets(obs, actions)
        critic_loss = nn.functional.mse_loss(q1, target_q) + nn.functional.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ── Actor 업데이트 ───────────────────────────────────────────────
        new_action, log_prob, _ = self.policy(obs)
        q1_new, q2_new = self.q_nets(obs, new_action)
        q_min = torch.min(q1_new, q2_new)
        actor_loss = (alpha * log_prob - q_min).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ── α 업데이트 ───────────────────────────────────────────────────
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ── Target soft update ───────────────────────────────────────────
        for p, p_target in zip(self.q_nets.parameters(), self.q_target.parameters()):
            p_target.data.mul_(1.0 - self.tau).add_(p.data * self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "q_mean": q1.mean().item(),
        }

    def save(self, path: str) -> None:
        torch.save({
            "policy": self.policy.state_dict(),
            "q_nets": self.q_nets.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }, path)
        print(f"모델 저장: {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.policy.load_state_dict(ckpt["policy"])
        self.q_nets.load_state_dict(ckpt["q_nets"])
        self.q_target.load_state_dict(ckpt["q_nets"])
        self.log_alpha = torch.tensor(
            ckpt["log_alpha"].item(), dtype=torch.float32,
            device=self.device, requires_grad=False,
        )
        self.policy.eval()
        self.q_nets.eval()
        print(f"모델 로드: {path}")
