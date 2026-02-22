"""
pg_agent.py - REINFORCE (Vanilla Policy Gradient)

PPO와 비교하기 위한 기본 Policy Gradient 구현.
PPO의 출발점이 된 알고리즘이며, PPO가 해결하는 문제를 직접 확인할 수 있다.

[REINFORCE vs PPO 핵심 차이]

  REINFORCE                         PPO
  ─────────────────────────────     ──────────────────────────────
  에피소드 1개 수집                  2048 스텝 수집
  G_t (Monte Carlo return) 사용     GAE 어드밴티지 사용
  업데이트 크기 제한 없음            Clip으로 최대 ±20% 제한
  데이터 1번 사용 후 버림            같은 데이터 10 epoch 재사용
  Critic 없음                       Actor + Critic

[REINFORCE 알고리즘]
  for each episode:
    1. 에피소드 완주: (s_0, a_0, r_0), (s_1, a_1, r_1), ..., (s_T, a_T, r_T)
    2. G_t 계산 (끝에서부터 역방향):
         G_T = r_T
         G_t = r_t + γ · G_{t+1}
    3. 손실 계산:
         L = -sum( log π(a_t|s_t) · G_t )
    4. 신경망 1번 업데이트

[예상되는 문제]
  - 학습 곡선이 위아래로 크게 흔들림 (고분산)
  - 어렵게 올라간 성능이 갑자기 무너지는 현상 (치명적 업데이트)
  - PPO 대비 훨씬 느린 수렴
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam


class PolicyNet(nn.Module):
    """
    REINFORCE용 단순 정책 신경망.

    PPO의 Actor-Critic과 달리 Actor(정책)만 있다.
    Critic(가치 추정)이 없으므로 GAE를 사용할 수 없다.
    대신 에피소드 전체 누적 보상 G_t를 직접 사용한다.

    구조: Linear(obs_dim→64) → Tanh → Linear(64→64) → Tanh → Linear(64→action_dim)
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        # Orthogonal 초기화 (PPO와 동일 조건으로 비교)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=(2 ** 0.5))
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> Categorical:
        logits = self.net(obs)
        return Categorical(logits=logits)


class REINFORCEAgent:
    """
    REINFORCE (Vanilla Policy Gradient) 에이전트.

    Args:
        obs_dim    : 관측 차원
        action_dim : 행동 개수
        hidden_dim : 은닉층 크기
        lr         : 학습률
        gamma      : 할인율
        device     : 디바이스
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: torch.device = torch.device("cpu"),
    ):
        self.gamma = gamma
        self.device = device

        self.policy = PolicyNet(obs_dim, action_dim, hidden_dim).to(device)
        self.optimizer = Adam(self.policy.parameters(), lr=lr)

        # 에피소드 동안 수집한 데이터 (update() 호출 시 소비됨)
        self._log_probs: list[torch.Tensor] = []
        self._rewards: list[float] = []

    def select_action(self, obs_np) -> int:
        """
        현재 정책으로 행동을 선택하고 log_prob을 내부에 저장한다.

        REINFORCE는 에피소드가 끝난 후 한꺼번에 업데이트하므로
        log_prob을 리스트에 계속 쌓아둔다.
        """
        obs_tensor = torch.tensor(obs_np, dtype=torch.float32).to(self.device)
        dist = self.policy(obs_tensor)
        action = dist.sample()
        self._log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward: float) -> None:
        """한 스텝의 보상을 저장한다."""
        self._rewards.append(reward)

    def update(self) -> dict:
        """
        에피소드 종료 후 REINFORCE 업데이트를 수행한다.

        [G_t 계산 - 역방향 누적]
          G_T = r_T
          G_t = r_t + γ · G_{t+1}

          예시 (γ=0.99, 3스텝 에피소드):
            r = [1, 1, 1]
            G_2 = 1
            G_1 = 1 + 0.99·1 = 1.99
            G_0 = 1 + 0.99·1.99 = 2.97

        [손실 계산]
          L = -sum( log π(a_t|s_t) · G_t )
            = -(log π(a_0|s_0)·G_0 + log π(a_1|s_1)·G_1 + ...)

          G_t가 클수록 해당 행동을 더 강하게 강화한다.

        [PPO와의 핵심 차이]
          - G_t 정규화는 하지만 clip이 없음 → 큰 G_t는 큰 업데이트로 이어짐
          - 1번만 업데이트하고 데이터 버림
        """
        T = len(self._rewards)

        # ── G_t 계산 (역방향) ─────────────────────────────────────────────
        returns = torch.zeros(T, device=self.device)
        G = 0.0
        for t in reversed(range(T)):
            G = self._rewards[t] + self.gamma * G
            returns[t] = G

        # ── G_t 정규화 (분산 감소를 위한 실용적 트릭) ─────────────────────
        # 주의: 정규화를 해도 PPO의 Clip만큼 안정적이지 않다.
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # ── 손실 계산 ──────────────────────────────────────────────────────
        log_probs = torch.stack(self._log_probs)  # (T,)

        # log_prob * G_t: G_t > 0 이면 이 행동의 확률을 높임
        #                 G_t < 0 이면 이 행동의 확률을 낮춤
        # 부호 반전: 최대화 문제를 최소화 문제로 변환
        loss = -(log_probs * returns).sum()

        # ── 업데이트 ───────────────────────────────────────────────────────
        # PPO와 달리 clip 없이 한 번만 업데이트
        # 큰 G_t가 있으면 그대로 큰 gradient가 흐른다 → 불안정의 원인
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 내부 버퍼 초기화 (에피소드 데이터 버림)
        ep_reward = sum(self._rewards)
        self._log_probs = []
        self._rewards = []

        return {"loss": loss.item(), "ep_reward": ep_reward, "ep_length": T}
