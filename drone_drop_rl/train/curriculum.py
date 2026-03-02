"""
curriculum.py — 커리큘럼 학습 매니저

3단계 난이도 커리큘럼:
  Easy   → 큰 타겟, 바람 없음, 가까운 출발
  Medium → 보통 타겟, 약한 바람, 중간 거리
  Hard   → 작은 타겟, 강한 바람, 먼 거리

승급 조건: 최근 100 에피소드 성공률 80% 이상
"""

from collections import deque
import numpy as np


STAGES = [
    {
        "name": "Easy",
        "target_radius": 10.0,    # 성공 판정 반경 (m)
        "max_wind": 0.0,          # 바람 없음
        "start_dist_range": (20.0, 40.0),   # 시작 거리 (m)
        "world_size": 100.0,
    },
    {
        "name": "Medium",
        "target_radius": 5.0,
        "max_wind": 2.0,
        "start_dist_range": (40.0, 100.0),
        "world_size": 150.0,
    },
    {
        "name": "Hard",
        "target_radius": 3.0,
        "max_wind": 5.0,
        "start_dist_range": (50.0, 150.0),
        "world_size": 200.0,
    },
]


class CurriculumManager:
    """커리큘럼 학습 단계 관리."""

    def __init__(
        self,
        stages: list = None,
        promotion_window: int = 100,
        promotion_threshold: float = 0.8,
    ):
        self.stages = stages or STAGES
        self.promotion_window = promotion_window
        self.promotion_threshold = promotion_threshold
        self.current_stage_idx = 0
        self._success_history = deque(maxlen=promotion_window)

    @property
    def current_stage(self) -> dict:
        return self.stages[self.current_stage_idx]

    @property
    def stage_name(self) -> str:
        return self.current_stage["name"]

    @property
    def is_final_stage(self) -> bool:
        return self.current_stage_idx >= len(self.stages) - 1

    @property
    def success_rate(self) -> float:
        if len(self._success_history) == 0:
            return 0.0
        return np.mean(self._success_history)

    def record_episode(self, landing_distance: float) -> bool:
        """
        에피소드 결과를 기록하고, 승급 여부를 반환한다.

        Args:
            landing_distance: 착지 거리 (m). None이면 실패.

        Returns:
            promoted: 승급했으면 True
        """
        target_radius = self.current_stage["target_radius"]
        success = (
            landing_distance is not None
            and landing_distance < target_radius
        )
        self._success_history.append(float(success))

        # 승급 체크
        if (
            not self.is_final_stage
            and len(self._success_history) >= self.promotion_window
            and self.success_rate >= self.promotion_threshold
        ):
            self.current_stage_idx += 1
            self._success_history.clear()
            return True

        return False

    def get_env_kwargs(self) -> dict:
        """현재 단계에 맞는 환경 파라미터를 반환한다."""
        stage = self.current_stage
        return {
            "max_wind": stage["max_wind"],
            "world_size": stage["world_size"],
        }

    def summary(self) -> str:
        stage = self.current_stage
        return (
            f"[{stage['name']}] "
            f"반경={stage['target_radius']}m | "
            f"바람≤{stage['max_wind']}m/s | "
            f"성공률={self.success_rate:.1%} "
            f"({len(self._success_history)}/{self.promotion_window})"
        )
