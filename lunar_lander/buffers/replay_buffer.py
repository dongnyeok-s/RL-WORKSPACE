"""
replay_buffer.py — Off-Policy Replay Buffer

DQN과 SAC가 공용으로 사용하는 경험 리플레이 버퍼.

On-Policy 버퍼(RolloutBuffer)와의 핵심 차이:
  - GAE 계산 없음
  - 데이터가 업데이트 후에도 유지됨 (삭제하지 않음)
  - next_obs를 저장 (Bellman backup에 필요)
  - 랜덤 샘플링 (순차 순회 아님)
"""

import numpy as np
import torch


class ReplayBuffer:

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int = 1,
        continuous: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        self.capacity = capacity
        self.device = device
        self._ptr = 0
        self._size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        if continuous:
            self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        else:
            self.actions = np.zeros(capacity, dtype=np.int64)
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
            "actions": torch.tensor(self.actions[indices]).to(self.device),
            "rewards": torch.tensor(self.rewards[indices], dtype=torch.float32).to(self.device),
            "next_obs": torch.tensor(self.next_obs[indices], dtype=torch.float32).to(self.device),
            "dones": torch.tensor(self.dones[indices], dtype=torch.float32).to(self.device),
        }

    def __len__(self):
        return self._size
