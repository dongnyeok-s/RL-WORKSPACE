"""
rollout_buffer.py — On-Policy Rollout Buffer + GAE

PPO용 버퍼. 기존 cartpole/buffer.py 기반에 continuous action 저장 지원 추가.

[GAE]
  δ_t = r_t + γ·V(s_{t+1})·(1-done) - V(s_t)
  A_t = δ_t + (γλ)·A_{t+1}·(1-done)
  R_t = A_t + V(s_t)
"""

import numpy as np
import torch
from typing import Iterator


class RolloutBuffer:

    def __init__(
        self,
        rollout_length: int,
        obs_dim: int,
        action_dim: int = 1,
        continuous: bool = False,
        gamma: float = 0.99,
        lam: float = 0.95,
        device: torch.device = torch.device("cpu"),
    ):
        self.rollout_length = rollout_length
        self.obs_dim = obs_dim
        self.continuous = continuous
        self.gamma = gamma
        self.lam = lam
        self.device = device

        self._ptr = 0
        self._gae_computed = False

        self.obs = np.zeros((rollout_length, obs_dim), dtype=np.float32)
        if continuous:
            self.actions = np.zeros((rollout_length, action_dim), dtype=np.float32)
        else:
            self.actions = np.zeros(rollout_length, dtype=np.int64)
        self.log_probs = np.zeros(rollout_length, dtype=np.float32)
        self.rewards = np.zeros(rollout_length, dtype=np.float32)
        self.values = np.zeros(rollout_length, dtype=np.float32)
        self.dones = np.zeros(rollout_length, dtype=np.float32)

        self.advantages = np.zeros(rollout_length, dtype=np.float32)
        self.returns = np.zeros(rollout_length, dtype=np.float32)

    def store(self, obs, action, log_prob, reward, value, done):
        assert self._ptr < self.rollout_length
        self.obs[self._ptr] = obs
        self.actions[self._ptr] = action
        self.log_probs[self._ptr] = log_prob
        self.rewards[self._ptr] = reward
        self.values[self._ptr] = value
        self.dones[self._ptr] = float(done)
        self._ptr += 1

    def compute_gae(self, last_value: float, last_done: bool) -> None:
        T = self.rollout_length
        gae = 0.0
        next_value = last_value * (1.0 - float(last_done))

        for t in reversed(range(T)):
            if t == T - 1:
                next_v = next_value
            else:
                next_v = self.values[t + 1]

            delta = self.rewards[t] + self.gamma * next_v * (1.0 - self.dones[t]) - self.values[t]
            gae = delta + self.gamma * self.lam * (1.0 - self.dones[t]) * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std() + 1e-8
        self.advantages = (self.advantages - adv_mean) / adv_std

        self._gae_computed = True

    def get_minibatches(self, batch_size: int) -> Iterator[dict]:
        assert self._gae_computed
        T = self.rollout_length
        indices = np.random.permutation(T)

        action_dtype = torch.float32 if self.continuous else torch.long

        for start in range(0, T, batch_size):
            idx = indices[start : start + batch_size]
            yield {
                "obs": torch.tensor(self.obs[idx], dtype=torch.float32).to(self.device),
                "actions": torch.tensor(self.actions[idx], dtype=action_dtype).to(self.device),
                "log_probs": torch.tensor(self.log_probs[idx], dtype=torch.float32).to(self.device),
                "advantages": torch.tensor(self.advantages[idx], dtype=torch.float32).to(self.device),
                "returns": torch.tensor(self.returns[idx], dtype=torch.float32).to(self.device),
            }

    def reset(self):
        self._ptr = 0
        self._gae_computed = False

    def is_full(self):
        return self._ptr >= self.rollout_length
