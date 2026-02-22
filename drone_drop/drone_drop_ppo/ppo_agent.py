"""
ppo_agent.py - PPO (Proximal Policy Optimization) 업데이트 로직

[PPO의 핵심 아이디어]
기존 Policy Gradient의 문제:
  - 한 번의 큰 업데이트가 정책을 망칠 수 있다.
  - on-policy 특성상 데이터를 한 번밖에 쓸 수 없어 비효율적이다.

PPO의 해결책:
  1. Clipped Surrogate Objective:
     정책 비율(ratio)이 [1-ε, 1+ε] 범위를 벗어나면 gradient를 차단해
     업데이트 크기를 제한한다.

  2. Multiple Epochs:
     같은 rollout 데이터를 n_epochs번 재사용해 학습 효율을 높인다.
     (단, clip 덕분에 정책이 너무 많이 바뀌는 것은 방지됨)

[PPO Clip Loss 수식]
  정책 비율:
    r_t(θ) = π_θ(a|s) / π_θ_old(a|s)
           = exp(log π_θ(a|s) - log π_θ_old(a|s))

  Clipped Surrogate (최대화 목표):
    L^CLIP = E_t [ min(r_t · Â_t,  clip(r_t, 1-ε, 1+ε) · Â_t) ]

  Critic (MSE 손실):
    L^VF = E_t [ (V_θ(s_t) - R_t)² ]

  Entropy Bonus (탐험 장려):
    L^S = E_t [ H[π_θ(·|s_t)] ]

  최종 손실 (최소화 형태):
    L = -L^CLIP + c1 · L^VF - c2 · L^S
"""

import torch
import torch.nn as nn
from torch.optim import Adam

from .network import ActorCritic
from .buffer import RolloutBuffer


class PPOAgent:
    """
    PPO 에이전트.

    Args:
        obs_dim       : 관측 차원 (CartPole = 4)
        action_dim    : 이산 행동 개수 (CartPole = 2)
        hidden_dim    : Actor-Critic 은닉층 크기 (기본 64)
        lr            : Adam 학습률 (기본 3e-4)
        clip_eps      : PPO clip 범위 ε (기본 0.2)
        value_coef    : Critic 손실 가중치 c1 (기본 0.5)
        entropy_coef  : 엔트로피 보너스 가중치 c2 (기본 0.01)
        max_grad_norm : 그래디언트 클리핑 최대 norm (기본 0.5)
        device        : 학습 디바이스
    """

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
        device: torch.device = torch.device("cpu"),
    ):
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        # Actor-Critic 신경망
        self.network = ActorCritic(obs_dim, action_dim, hidden_dim).to(device)

        # Adam 옵티마이저
        self.optimizer = Adam(self.network.parameters(), lr=lr)

    def select_action(self, obs_np) -> tuple[int, float, float]:
        """
        환경과 상호작용할 때 행동을 선택한다.

        Args:
            obs_np: numpy 배열 형태의 관측 (obs_dim,)

        Returns:
            action   : 선택된 행동 (int)
            log_prob : log π_old(a|s) (float) → 버퍼에 저장
            value    : V(s) 추정값 (float) → 버퍼에 저장
        """
        obs_tensor = torch.tensor(obs_np, dtype=torch.float32).to(self.device)
        return self.network.get_action(obs_tensor)

    @torch.no_grad()
    def get_value(self, obs_np) -> float:
        """
        rollout 완료 후 마지막 상태의 V(s_T)를 계산한다.
        GAE bootstrap에 사용.
        """
        obs_tensor = torch.tensor(obs_np, dtype=torch.float32).to(self.device)
        _, value = self.network(obs_tensor)
        return value.squeeze().item()

    def update(
        self, buffer: RolloutBuffer, n_epochs: int, batch_size: int
    ) -> dict:
        """
        PPO 업데이트: 버퍼 데이터로 n_epochs 반복 학습.

        [업데이트 플로우]
        for epoch in range(n_epochs):
          for minibatch in buffer.get_minibatches(batch_size):
            1. 현재 정책으로 (log_prob_new, value_new, entropy) 재계산
            2. ratio = exp(log_prob_new - log_prob_old)
            3. Clip Loss 계산
            4. Critic Loss 계산
            5. Entropy Bonus 계산
            6. 합산 후 backward → grad clip → step

        Args:
            buffer   : GAE가 계산된 RolloutBuffer
            n_epochs : 업데이트 반복 횟수 (예: 10)
            batch_size: 미니배치 크기 (예: 64)

        Returns:
            dict: 평균 손실값들 (로깅용)
              - actor_loss   : PPO clip loss
              - critic_loss  : MSE value loss
              - entropy      : 평균 엔트로피 (높을수록 탐험 많이 함)
              - approx_kl    : 근사 KL divergence (정상 범위: 0.01~0.03)
        """
        self.network.train()

        # 여러 에포크의 평균 손실을 추적
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        n_updates = 0

        for _ in range(n_epochs):
            for batch in buffer.get_minibatches(batch_size):
                obs = batch["obs"]               # (B, obs_dim)
                actions = batch["actions"]       # (B,)
                old_log_probs = batch["log_probs"]  # (B,) → log π_old(a|s)
                advantages = batch["advantages"] # (B,) → 정규화된 GAE
                returns = batch["returns"]       # (B,) → Critic 타겟

                # ── 현재 정책으로 재평가 ──────────────────────────────────
                new_log_probs, values, entropy = self.network.evaluate_actions(obs, actions)

                # ── PPO Clip Loss ─────────────────────────────────────────
                # 정책 비율: 현재 정책 / 이전 정책
                # log 공간에서 계산 → 수치 안정성
                ratio = torch.exp(new_log_probs - old_log_probs)

                # clip 전 목표: ratio * advantage
                surr1 = ratio * advantages

                # clip 후 목표: clip(ratio, 1-ε, 1+ε) * advantage
                # advantage > 0이면: 좋은 행동이므로 ratio가 (1+ε) 이상 커지는 걸 막음
                # advantage < 0이면: 나쁜 행동이므로 ratio가 (1-ε) 이하 작아지는 걸 막음
                surr2 = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages

                # 두 목표 중 더 보수적인(작은) 값을 선택 → 최소화를 위해 부호 반전
                actor_loss = -torch.min(surr1, surr2).mean()

                # ── Critic Loss (MSE) ─────────────────────────────────────
                # V(s)가 TD-λ returns R_t에 가까워지도록 학습
                critic_loss = nn.functional.mse_loss(values, returns)

                # ── Entropy Bonus ─────────────────────────────────────────
                # 엔트로피가 클수록 탐험을 많이 한다.
                # entropy_coef로 탐험과 활용의 균형 조절
                entropy_loss = -entropy.mean()  # 최대화 → 부호 반전

                # ── 최종 손실 ─────────────────────────────────────────────
                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    + self.entropy_coef * entropy_loss
                )

                # ── 역전파 + 그래디언트 클리핑 ───────────────────────────
                self.optimizer.zero_grad()
                loss.backward()
                # 그래디언트 클리핑: 너무 큰 그래디언트로 인한 불안정 학습 방지
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # ── 모니터링 지표 집계 ────────────────────────────────────
                # Approximate KL Divergence: 정책 변화량 모니터링
                # 정상 범위: 0.01 ~ 0.03. 너무 크면 학습 불안정 신호
                with torch.no_grad():
                    approx_kl = (old_log_probs - new_log_probs).mean().item()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += (-entropy_loss).item()  # 원래 엔트로피 값
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
        """학습된 모델 가중치를 저장한다."""
        torch.save(self.network.state_dict(), path)
        print(f"모델 저장: {path}")

    def load(self, path: str) -> None:
        """저장된 모델 가중치를 불러온다."""
        self.network.load_state_dict(torch.load(path, map_location=self.device))
        self.network.eval()
        print(f"모델 로드: {path}")
