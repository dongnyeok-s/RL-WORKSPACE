"""
network.py - Actor-Critic 신경망

PPO는 한 신경망이 두 가지 역할을 동시에 수행한다.
  - Actor  : 상태(s)를 보고 어떤 행동을 할지 확률 분포 π(a|s)를 출력
  - Critic : 상태(s)를 보고 그 상태의 가치 V(s)를 추정

[공유 Backbone 구조]
                   obs (4차원)
                       │
         ┌─────────────▼─────────────┐
         │  Backbone: Linear → Tanh  │  ← obs 특징 추출 (공유)
         │  Linear(4→64) → Tanh      │
         │  Linear(64→64) → Tanh     │
         └──────────┬────────────────┘
                    │
          ┌─────────┴─────────┐
          │                   │
    Actor Head           Critic Head
  Linear(64→2)          Linear(64→1)
  Categorical 분포        스칼라 V(s)
  π(a|s)

CartPole-v1:
  - obs_dim    = 4  (카트 위치, 속도, 폴 각도, 각속도)
  - action_dim = 2  (왼쪽, 오른쪽)
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


def _init_weights(layer: nn.Linear, gain: float) -> nn.Linear:
    """
    Orthogonal 초기화 - PPO 논문 권장 방식.

    gain 값에 따른 역할:
      - gain = sqrt(2) : backbone 레이어 → 충분한 그래디언트 흐름 보장
      - gain = 0.01    : actor head → 초기 정책을 균등 분포에 가깝게 만듦
      - gain = 1.0     : critic head → 중립적 가치 추정으로 시작
    """
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    """
    PPO Actor-Critic 신경망.

    Args:
        obs_dim    : 관측 차원 (CartPole = 4)
        action_dim : 이산 행동 개수 (CartPole = 2)
        hidden_dim : 은닉층 뉴런 수 (기본값 64)
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()

        # ── Shared Backbone ──────────────────────────────────────────────
        # Actor와 Critic이 같은 특징 표현을 공유한다.
        # CartPole처럼 저차원 환경에서는 공유 구조가 수렴 속도에 유리하다.
        # Activation으로 Tanh를 사용: 관측값이 bounded 범위를 가지므로 수치 안정성이 좋음
        self.backbone = nn.Sequential(
            _init_weights(nn.Linear(obs_dim, hidden_dim), gain=(2 ** 0.5)),
            nn.Tanh(),
            _init_weights(nn.Linear(hidden_dim, hidden_dim), gain=(2 ** 0.5)),
            nn.Tanh(),
        )

        # ── Actor Head ───────────────────────────────────────────────────
        # 행동 로짓(logit)을 출력한다. Softmax는 Categorical 내부에서 처리.
        # gain=0.01로 초기화 → 초기 정책이 모든 행동에 거의 균등한 확률을 가짐
        self.actor_head = _init_weights(nn.Linear(hidden_dim, action_dim), gain=0.01)

        # ── Critic Head ──────────────────────────────────────────────────
        # 상태 가치 V(s)를 스칼라로 출력한다.
        self.critic_head = _init_weights(nn.Linear(hidden_dim, 1), gain=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        """
        순전파: 관측 → (행동 분포, 상태 가치)

        Args:
            obs: (batch, obs_dim) 또는 (obs_dim,) 형태의 관측 텐서

        Returns:
            dist : Categorical 분포 객체 → .sample(), .log_prob() 사용 가능
            value: (batch, 1) 또는 (1,) 형태의 상태 가치 V(s)
        """
        features = self.backbone(obs)
        logits = self.actor_head(features)   # 행동 로짓 (batch, action_dim)
        value = self.critic_head(features)   # 상태 가치 (batch, 1)
        dist = Categorical(logits=logits)    # 내부적으로 softmax 처리
        return dist, value

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor) -> tuple[int, float, float]:
        """
        환경과 상호작용할 때 사용. (학습 그래디언트 추적 없음 → 빠름)

        Args:
            obs: (obs_dim,) 단일 관측 텐서

        Returns:
            action   : 선택된 행동 (int)
            log_prob : 해당 행동의 log 확률 (float) → rollout buffer에 저장
            value    : Critic이 추정한 V(s) (float) → GAE 계산에 사용
        """
        dist, value = self.forward(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.squeeze().item()

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        PPO 업데이트 시 사용: 저장된 (obs, action)을 현재 정책으로 재평가.

        Args:
            obs    : (batch, obs_dim) 미니배치 관측
            actions: (batch,) 미니배치 행동

        Returns:
            log_prob: (batch,) 현재 정책 π_θ(a|s)의 log 확률
                      → PPO ratio = exp(log_prob_new - log_prob_old) 계산에 사용
            value   : (batch,) 현재 Critic의 V(s) 추정
            entropy : (batch,) 행동 분포의 엔트로피
                      → 높을수록 탐험을 많이 한다는 의미 (entropy bonus로 탐험 장려)
        """
        dist, value = self.forward(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, value.squeeze(-1), entropy
