"""
rewards.py — 4계층 계층적 보상 함수 + Sparse 보상 (비교 실험용)

Shaped Reward (4계층):
  Layer 1 — 안전:     경계 위반 시 큰 음의 보상 + 에피소드 종료
  Layer 2 — 효율성:   시간 페널티 + 에너지 페널티 (매 스텝)
  Layer 3 — 접근 유도: Potential-based shaping + 정렬 보너스 (매 스텝, 투하 전)
  Layer 4 — 투하 정밀도: 거리 비례 보상 + 정밀도 티어 보너스 (투하 시 1회)

Sparse Reward (대조군):
  투하 시 거리 기반 보상만 부여, 중간 과정 보상 없음

보상 해킹 방지:
  1. Potential-based shaping (Ng et al. 1999) → 최적 정책 불변 수학적 보장
  2. 에너지 페널티 → 타겟 근처 진동(jittering) 방지
  3. 1회성 투하 → 반복 투하 시도 불가
"""

import numpy as np


class ShapedReward:
    """4계층 계층적 보상 함수."""

    # ── 안전 경계 ──────────────────────────────────────────────────────────────
    SAFETY_MIN_ALT = 5.0       # 최소 안전 고도 (m)
    SAFETY_MAX_SPEED = 30.0    # 최대 안전 속도 (m/s)
    SOFT_MIN_ALT = 15.0        # 소프트 고도 경계 (m) — 경고 구간

    # ── 보상 스케일 ────────────────────────────────────────────────────────────
    TIME_PENALTY = -0.01       # 매 스텝 시간 페널티
    ENERGY_COEFF = 0.005       # 에너지 페널티 계수
    APPROACH_SCALE = 3.0       # 접근 보상 스케일
    ALIGNMENT_BONUS = 0.02     # 정렬 보너스 최대값
    TIMING_BONUS = 0.1         # 타이밍 보상
    TIMING_THRESHOLD = 0.05    # 타이밍 임계값

    # ── 투하 보상 ──────────────────────────────────────────────────────────────
    MAX_REWARD_DIST = 50.0     # 보상이 0이 되는 최대 거리 (m)
    DISTANCE_SCALE = 15.0      # 거리 비례 보상 최대값

    # 정밀도 티어: (거리 임계값, 보너스)
    PRECISION_TIERS = [
        (1.0, 10.0),   # 1m 이내: +10
        (3.0, 5.0),    # 3m 이내: +5
        (5.0, 2.0),    # 5m 이내: +2
    ]

    # ── 페널티 ─────────────────────────────────────────────────────────────────
    SAFETY_VIOLATION_PENALTY = -10.0
    SPEED_VIOLATION_PENALTY = -8.0
    OOB_PENALTY = -5.0        # Out-of-bounds
    TIMEOUT_PENALTY = -3.0    # 시간 초과

    def __init__(self, gamma: float = 0.99, max_approach_dist: float = 150.0,
                 max_lateral: float = 50.0):
        self.gamma = gamma
        self.max_approach_dist = max_approach_dist
        self.max_lateral = max_lateral
        self._prev_target_dist = None

    def reset(self, initial_target_dist: float):
        """에피소드 시작 시 호출."""
        self._prev_target_dist = initial_target_dist

    def compute(self, state: dict, action: np.ndarray, info: dict) -> tuple:
        """
        보상을 계산한다.

        Args:
            state: 현재 상태 정보
                - altitude: 현재 고도 (m)
                - speed: 현재 속도 (m/s)
                - target_dist: 타겟까지 수평 거리 (m)
                - lateral_offset: 타겟 대비 횡방향 오프셋 (m)
                - timing_signal: 투하 타이밍 신호 (0 = 최적)
            action: 에이전트 액션 [lateral, forward, vertical, drop]
            info: 추가 정보
                - has_dropped: 이미 투하했는지
                - landing_distance: 착지 거리 (투하 시)
                - oob: 영역 이탈 여부
                - timeout: 시간 초과 여부

        Returns:
            (reward, terminated, truncated)
        """
        reward = 0.0

        # ═══ Layer 1: 안전 (최우선) ═══════════════════════════════════════════
        if state["altitude"] < self.SAFETY_MIN_ALT:
            return self.SAFETY_VIOLATION_PENALTY, True, False

        if state["speed"] > self.SAFETY_MAX_SPEED:
            return self.SPEED_VIOLATION_PENALTY, True, False

        # 소프트 고도 페널티 (경고 구간)
        if state["altitude"] < self.SOFT_MIN_ALT:
            ratio = (self.SOFT_MIN_ALT - state["altitude"]) / self.SOFT_MIN_ALT
            reward -= 0.5 * ratio

        # ═══ Layer 2: 효율성 (매 스텝) ═══════════════════════════════════════
        reward += self.TIME_PENALTY
        reward -= self.ENERGY_COEFF * np.linalg.norm(action[:3])

        # ═══ Layer 3: 접근 유도 (투하 전) ════════════════════════════════════
        if not info["has_dropped"]:
            target_dist = state["target_dist"]

            # Potential-based shaping: γ·φ(s') - φ(s)
            phi_prev = -self._prev_target_dist / self.max_approach_dist
            phi_curr = -target_dist / self.max_approach_dist
            approach_reward = (self.gamma * phi_curr - phi_prev) * self.APPROACH_SCALE
            reward += approach_reward
            self._prev_target_dist = target_dist

            # 정렬 보너스: 타겟 바로 위 비행경로 유지
            alignment = 1.0 - min(
                abs(state["lateral_offset"]) / self.max_lateral, 1.0
            )
            reward += self.ALIGNMENT_BONUS * alignment

            # 타이밍 보상: 최적 투하 지점 근접 시
            if abs(state["timing_signal"]) < self.TIMING_THRESHOLD:
                reward += self.TIMING_BONUS

        # ═══ Layer 4: 투하 정밀도 (1회) ═══════════════════════════════════════
        if info.get("just_dropped", False):
            distance = info["landing_distance"]

            # 거리 비례 보상
            dist_reward = max(0.0, 1.0 - distance / self.MAX_REWARD_DIST) * self.DISTANCE_SCALE
            reward += dist_reward

            # 정밀도 티어 보너스
            for threshold, bonus in self.PRECISION_TIERS:
                if distance < threshold:
                    reward += bonus
                    break

            return reward, True, False

        # ═══ 에피소드 종료 조건 ══════════════════════════════════════════════
        if info.get("oob", False):
            return reward + self.OOB_PENALTY, False, True

        if info.get("timeout", False):
            return reward + self.TIMEOUT_PENALTY, False, True

        return reward, False, False


class SparseReward:
    """
    Sparse 보상 — 투하 결과에만 보상 부여 (대조 실험용).

    중간 과정에서는 보상 없음. 투하 시에만 거리 기반 보상.
    4계층 Shaped Reward와의 비교를 통해 보상 설계의 효과를 검증한다.
    """

    MAX_REWARD_DIST = 50.0
    DISTANCE_SCALE = 15.0
    PRECISION_TIERS = [
        (1.0, 10.0),
        (3.0, 5.0),
        (5.0, 2.0),
    ]
    OOB_PENALTY = -5.0
    TIMEOUT_PENALTY = -3.0
    SAFETY_MIN_ALT = 5.0
    SAFETY_MAX_SPEED = 30.0
    SAFETY_VIOLATION_PENALTY = -10.0
    SPEED_VIOLATION_PENALTY = -8.0

    def reset(self, initial_target_dist: float):
        pass

    def compute(self, state: dict, action: np.ndarray, info: dict) -> tuple:
        """Sparse: 투하 시에만 보상."""

        # 안전 위반은 동일하게 처리
        if state["altitude"] < self.SAFETY_MIN_ALT:
            return self.SAFETY_VIOLATION_PENALTY, True, False
        if state["speed"] > self.SAFETY_MAX_SPEED:
            return self.SPEED_VIOLATION_PENALTY, True, False

        # 투하 시: 거리 기반 보상만
        if info.get("just_dropped", False):
            distance = info["landing_distance"]
            reward = max(0.0, 1.0 - distance / self.MAX_REWARD_DIST) * self.DISTANCE_SCALE
            for threshold, bonus in self.PRECISION_TIERS:
                if distance < threshold:
                    reward += bonus
                    break
            return reward, True, False

        if info.get("oob", False):
            return self.OOB_PENALTY, False, True
        if info.get("timeout", False):
            return self.TIMEOUT_PENALTY, False, True

        # 중간 과정: 보상 없음
        return 0.0, False, False
