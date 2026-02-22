"""
buffer.py - Rollout Buffer + GAE (Generalized Advantage Estimation)

PPO는 on-policy 알고리즘이다.
  → 현재 정책(π_old)으로 데이터를 수집하고
  → 같은 데이터를 여러 에포크(epoch) 재사용해 학습한다.
  → 학습이 끝나면 버퍼를 비우고 새로 수집한다.

[버퍼의 역할]
  1. rollout_length 스텝만큼 경험 저장
  2. GAE로 어드밴티지(A_t) 계산
  3. PPO 업데이트용 미니배치 제공

[GAE (Generalized Advantage Estimation)]
  기존 어드밴티지: A_t = R_t - V(s_t)  ← 분산이 크다
  TD(0) 어드밴티지: A_t = r_t + γV(s_{t+1}) - V(s_t)  ← 편향이 크다
  GAE는 이 둘의 bias-variance tradeoff를 λ로 조절한다.

  TD 오차:
    δ_t = r_t + γ · V(s_{t+1}) · (1 - done_t) - V(s_t)

  GAE 어드밴티지 (역방향 누적):
    A_t^GAE = δ_t + (γλ) · A_{t+1}^GAE · (1 - done_t)

    λ=0: TD(0), 편향 큼 / 분산 작음
    λ=1: Monte Carlo, 편향 작음 / 분산 큼
    λ=0.95: (권장) 두 극단의 균형점

  Critic 학습 타겟 (TD-lambda returns):
    R_t = A_t^GAE + V(s_t)
"""

import numpy as np
import torch
from typing import Iterator


class RolloutBuffer:
    """
    PPO 롤아웃 버퍼.

    Args:
        rollout_length : 한 번의 rollout에서 수집할 스텝 수 (예: 2048)
        obs_dim        : 관측 차원 (CartPole = 4)
        gamma          : 할인율 γ (기본 0.99)
        lam            : GAE lambda λ (기본 0.95)
        device         : 학습에 사용할 디바이스
    """

    def __init__(
        self,
        rollout_length: int,
        obs_dim: int,
        gamma: float = 0.99,
        lam: float = 0.95,
        device: torch.device = torch.device("cpu"),
    ):
        self.rollout_length = rollout_length
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.lam = lam
        self.device = device

        self._ptr = 0  # 현재 저장 위치
        self._gae_computed = False

        # ── 데이터 저장 배열 (numpy, 나중에 tensor로 변환) ───────────────
        self.obs = np.zeros((rollout_length, obs_dim), dtype=np.float32)
        self.actions = np.zeros(rollout_length, dtype=np.int64)
        self.log_probs = np.zeros(rollout_length, dtype=np.float32)
        self.rewards = np.zeros(rollout_length, dtype=np.float32)
        self.values = np.zeros(rollout_length, dtype=np.float32)
        self.dones = np.zeros(rollout_length, dtype=np.float32)

        # compute_gae() 호출 후 채워지는 배열
        self.advantages = np.zeros(rollout_length, dtype=np.float32)
        self.returns = np.zeros(rollout_length, dtype=np.float32)

    def store(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        """
        한 스텝의 경험을 버퍼에 저장한다.

        Args:
            obs      : 현재 관측 (obs_dim,)
            action   : 선택한 행동
            log_prob : log π_old(a|s) → PPO ratio 계산에 필요
            reward   : 환경으로부터 받은 보상
            value    : Critic의 V(s) 추정값 → GAE 계산에 필요
            done     : 에피소드 종료 여부
        """
        assert self._ptr < self.rollout_length, "버퍼가 가득 찼습니다. reset() 을 먼저 호출하세요."
        self.obs[self._ptr] = obs
        self.actions[self._ptr] = action
        self.log_probs[self._ptr] = log_prob
        self.rewards[self._ptr] = reward
        self.values[self._ptr] = value
        self.dones[self._ptr] = float(done)
        self._ptr += 1

    def compute_gae(self, last_value: float, last_done: bool) -> None:
        """
        GAE 어드밴티지와 TD-λ returns를 계산한다.
        반드시 롤아웃 수집이 완료된 후 호출해야 한다.

        Args:
            last_value : 롤아웃 마지막 다음 상태의 V(s_T)
                         에피소드가 중간에 끊겼을 때 bootstrap에 사용.
            last_done  : 마지막 상태가 에피소드 종료였는지 여부

        [핵심 구현: 역방향 누적]
        마지막 스텝부터 거꾸로 순회하면서 GAE를 누적한다.
        done=True인 스텝에서 gae를 0으로 리셋 (에피소드 경계)

          t=T-1:  gae = δ_{T-1}
          t=T-2:  gae = δ_{T-2} + γλ · gae_{T-1} · (1 - done_{T-2})
          ...
        """
        T = self.rollout_length
        gae = 0.0

        # 마지막 스텝 이후 next_value: 에피소드가 끝나지 않았으면 bootstrap
        next_value = last_value * (1.0 - float(last_done))

        for t in reversed(range(T)):
            if t == T - 1:
                next_v = next_value
                next_done = float(last_done)
            else:
                next_v = self.values[t + 1]
                next_done = self.dones[t + 1]

            # TD 오차: δ_t = r_t + γ·V(s_{t+1})·(1-done_{t+1}) - V(s_t)
            # 주의: done은 "현재 스텝에서 에피소드가 끝났는가"
            # done=True이면 s_{t+1}은 새 에피소드 시작 → V(s_{t+1})을 0으로 처리
            delta = self.rewards[t] + self.gamma * next_v * (1.0 - self.dones[t]) - self.values[t]

            # GAE 누적: A_t = δ_t + γλ · A_{t+1} · (1-done_t)
            gae = delta + self.gamma * self.lam * (1.0 - self.dones[t]) * gae
            self.advantages[t] = gae

        # Critic 학습 타겟: R_t = A_t + V(s_t)
        self.returns = self.advantages + self.values

        # ── 어드밴티지 정규화 ─────────────────────────────────────────────
        # mean=0, std=1로 정규화하면 다양한 보상 스케일에 robust해진다.
        # 학습 안정성을 크게 향상시키는 실용적 트릭.
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std() + 1e-8  # 0 나누기 방지
        self.advantages = (self.advantages - adv_mean) / adv_std

        self._gae_computed = True

    def get_minibatches(self, batch_size: int) -> Iterator[dict]:
        """
        버퍼 전체 데이터를 shuffle 후 미니배치로 나눠 반환한다.
        PPO는 한 번의 rollout 데이터로 n_epochs 반복 학습하므로
        매 에포크마다 이 메서드를 호출해 다른 순서의 미니배치를 사용한다.

        Args:
            batch_size : 미니배치 크기 (예: 64)

        Yields:
            dict with keys:
              obs        : (batch, obs_dim) 텐서
              actions    : (batch,) 텐서
              log_probs  : (batch,) 텐서 → log π_old(a|s)
              advantages : (batch,) 텐서
              returns    : (batch,) 텐서 → Critic 학습 타겟
        """
        assert self._gae_computed, "compute_gae()를 먼저 호출하세요."
        T = self.rollout_length
        indices = np.random.permutation(T)  # 셔플

        for start in range(0, T, batch_size):
            batch_idx = indices[start : start + batch_size]
            yield {
                "obs": torch.tensor(self.obs[batch_idx], dtype=torch.float32).to(self.device),
                "actions": torch.tensor(self.actions[batch_idx], dtype=torch.long).to(self.device),
                "log_probs": torch.tensor(self.log_probs[batch_idx], dtype=torch.float32).to(self.device),
                "advantages": torch.tensor(self.advantages[batch_idx], dtype=torch.float32).to(self.device),
                "returns": torch.tensor(self.returns[batch_idx], dtype=torch.float32).to(self.device),
            }

    def reset(self) -> None:
        """버퍼를 초기화한다. 학습 완료 후 새 rollout 수집 전에 호출."""
        self._ptr = 0
        self._gae_computed = False

    def is_full(self) -> bool:
        """rollout_length만큼 데이터가 채워졌는지 확인."""
        return self._ptr >= self.rollout_length
