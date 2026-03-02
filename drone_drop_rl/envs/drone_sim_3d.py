"""
drone_sim_3d.py — 3D 드론 투하 시뮬레이션 Gymnasium 환경

drone_drop/env.py의 2D 설계를 3D로 확장한 환경.

[환경 설명]
드론이 3D 공간에서 타겟을 향해 이동하고, 적절한 타이밍에 물체를 투하하여
지상의 타겟에 최대한 가깝게 착지시킨다.

[핵심 물리]
  낙하 시간: t_fall = sqrt(2 * altitude / g)
  착지 위치: landing = drop_pos + velocity * t_fall + 0.5 * wind * t_fall²
  → 에이전트는 고도·속도·바람을 고려한 최적 투하 타이밍을 학습해야 한다.

[관측 공간] (14차원, 정규화)
   0- 2: 드론 위치 (x, y, z) / max_range
   3- 5: 드론 속도 (vx, vy, vz) / max_speed
   6- 8: 드론 자세 (roll, pitch, yaw) / pi
   9-10: 타겟 상대 위치 (dx, dz) / max_range
  11:    고도 / max_altitude
  12:    풍속 / max_wind
  13:    투하 타이밍 신호 (0 = 최적)

[행동 공간] (연속 4차원, [-1, 1])
  0: 좌우 속도 (x축)
  1: 전후 속도 (z축)
  2: 상하 속도 (y축)
  3: 투하 명령 (≥0이면 투하, 1회)

[보상]
  외부 RewardFunction 객체를 통해 계산 (rewards.py 참조)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .rewards import ShapedReward, SparseReward


class DroneDropEnv3D(gym.Env):
    """3D 드론 투하 시뮬레이션 환경."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode=None,
        reward_type: str = "shaped",
        world_size: float = 200.0,
        max_altitude: float = 80.0,
        default_altitude: float = 50.0,
        gravity: float = 9.81,
        dt: float = 1.0 / 30.0,
        max_steps: int = 600,
        max_speed: float = 20.0,
        max_wind: float = 5.0,
        gamma: float = 0.99,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.world_size = world_size
        self.max_altitude = max_altitude
        self.default_altitude = default_altitude
        self.gravity = gravity
        self.dt = dt
        self.max_steps = max_steps
        self.max_speed = max_speed
        self.max_wind = max_wind

        # 보상 함수 선택
        if reward_type == "shaped":
            self.reward_fn = ShapedReward(
                gamma=gamma,
                max_approach_dist=world_size,
                max_lateral=world_size * 0.5,
            )
        else:
            self.reward_fn = SparseReward()

        # 낙하 시간 (기본 고도 기준, 참고값)
        self.default_fall_time = (2.0 * default_altitude / gravity) ** 0.5

        # ── 관측/행동 공간 정의 ─────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(14,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32,
        )

        # ── 상태 변수 ──────────────────────────────────────────────────────
        self.drone_pos = np.zeros(3)    # [x, y, z] (y = 고도)
        self.drone_vel = np.zeros(3)    # [vx, vy, vz]
        self.drone_orient = np.zeros(3) # [roll, pitch, yaw] (단순화)
        self.target_pos = np.zeros(2)   # [x, z] (지상)
        self.wind_vel = np.zeros(2)     # [wx, wz] (수평 바람)
        self.has_dropped = False
        self.current_step = 0

        # 렌더링
        self.screen = None
        self.clock = None
        self.font = None

    def reset(self, seed=None, options=None):
        """환경 초기화."""
        super().reset(seed=seed)

        # 드론 시작 위치: 월드 가장자리에서 랜덤
        angle = self.np_random.uniform(0, 2 * np.pi)
        start_dist = self.np_random.uniform(
            self.world_size * 0.4, self.world_size * 0.7
        )
        self.drone_pos = np.array([
            start_dist * np.cos(angle),     # x
            self.default_altitude,          # y (고도)
            start_dist * np.sin(angle),     # z
        ])

        # 초기 속도: 타겟 방향으로 약간의 속도
        speed = self.np_random.uniform(3.0, 8.0)
        to_center = -self.drone_pos[[0, 2]]
        to_center_norm = to_center / (np.linalg.norm(to_center) + 1e-8)
        self.drone_vel = np.array([
            to_center_norm[0] * speed,
            0.0,
            to_center_norm[1] * speed,
        ])

        self.drone_orient = np.zeros(3)

        # 타겟: 중심 근처에 랜덤 배치
        self.target_pos = self.np_random.uniform(
            -self.world_size * 0.15, self.world_size * 0.15, size=2
        )

        # 바람: 랜덤 방향과 세기
        wind_speed = self.np_random.uniform(0, self.max_wind)
        wind_angle = self.np_random.uniform(0, 2 * np.pi)
        self.wind_vel = np.array([
            wind_speed * np.cos(wind_angle),
            wind_speed * np.sin(wind_angle),
        ])

        self.has_dropped = False
        self.current_step = 0

        # 보상 함수 초기화
        initial_dist = np.linalg.norm(
            self.drone_pos[[0, 2]] - self.target_pos
        )
        self.reward_fn.reset(initial_dist)

        return self._get_obs(), self._get_info()

    def _get_obs(self) -> np.ndarray:
        """14차원 정규화된 관측 벡터."""
        # 타겟까지 상대 위치
        dx = self.target_pos[0] - self.drone_pos[0]
        dz = self.target_pos[1] - self.drone_pos[2]

        # 투하 타이밍 신호 (drone_drop/env.py의 timing_signal 개념 확장)
        altitude = self.drone_pos[1]
        if altitude > 0:
            t_fall = (2.0 * altitude / self.gravity) ** 0.5
        else:
            t_fall = 0.0

        # 투하 시 예상 수평 이동 (드론 속도 + 바람 영향)
        drift_x = self.drone_vel[0] * t_fall + 0.5 * self.wind_vel[0] * t_fall ** 2
        drift_z = self.drone_vel[2] * t_fall + 0.5 * self.wind_vel[1] * t_fall ** 2

        # 타이밍 신호: 예상 착지 오차 / 최대 거리 (0 = 지금 투하하면 정확히 맞음)
        expected_landing_error = np.sqrt(
            (self.drone_pos[0] + drift_x - self.target_pos[0]) ** 2
            + (self.drone_pos[2] + drift_z - self.target_pos[1]) ** 2
        )
        timing_signal = expected_landing_error / self.world_size

        wind_magnitude = np.linalg.norm(self.wind_vel)

        return np.array([
            self.drone_pos[0] / self.world_size,
            self.drone_pos[1] / self.max_altitude,
            self.drone_pos[2] / self.world_size,
            self.drone_vel[0] / self.max_speed,
            self.drone_vel[1] / self.max_speed,
            self.drone_vel[2] / self.max_speed,
            self.drone_orient[0] / np.pi,
            self.drone_orient[1] / np.pi,
            self.drone_orient[2] / np.pi,
            dx / self.world_size,
            dz / self.world_size,
            altitude / self.max_altitude,
            wind_magnitude / self.max_wind,
            timing_signal,
        ], dtype=np.float32)

    def _compute_3d_landing(self) -> np.ndarray:
        """
        투하 시점에서 착지 위치를 해석적으로 계산한다.
        (drone_drop/env.py의 _compute_landing_x 확장)

        landing = drop_pos + vel * t_fall + 0.5 * wind * t_fall²
        """
        altitude = self.drone_pos[1]
        if altitude <= 0:
            return self.drone_pos[[0, 2]].copy()

        t_fall = (2.0 * altitude / self.gravity) ** 0.5

        landing_x = (
            self.drone_pos[0]
            + self.drone_vel[0] * t_fall
            + 0.5 * self.wind_vel[0] * t_fall ** 2
        )
        landing_z = (
            self.drone_pos[2]
            + self.drone_vel[2] * t_fall
            + 0.5 * self.wind_vel[1] * t_fall ** 2
        )

        return np.array([landing_x, landing_z])

    def step(self, action: np.ndarray):
        """
        한 스텝 진행.

        Args:
            action: [lateral_vel, forward_vel, vertical_vel, drop_cmd]
                    각 [-1, 1] 범위, 실제 속도로 매핑

        Returns:
            obs, reward, terminated, truncated, info
        """
        self.current_step += 1
        action = np.clip(action, -1.0, 1.0)

        # ── 1. 드론 이동 ──────────────────────────────────────────────────────
        # 액션을 실제 속도로 매핑
        target_vel = np.array([
            action[0] * self.max_speed * 0.5,   # 좌우
            action[2] * self.max_speed * 0.25,   # 상하 (더 제한적)
            action[1] * self.max_speed * 0.5,    # 전후
        ])

        # 부드러운 속도 변화 (관성)
        alpha = 0.3
        self.drone_vel = (1 - alpha) * self.drone_vel + alpha * target_vel

        # 바람 영향 (수평만)
        self.drone_vel[0] += self.wind_vel[0] * self.dt * 0.1
        self.drone_vel[2] += self.wind_vel[1] * self.dt * 0.1

        # 위치 업데이트
        self.drone_pos += self.drone_vel * self.dt

        # 고도 제한 (지면 아래로 내려가지 않음)
        self.drone_pos[1] = max(0.0, self.drone_pos[1])

        # 자세 업데이트 (단순화: 속도 방향으로 약간 기울어짐)
        speed_h = np.linalg.norm(self.drone_vel[[0, 2]])
        if speed_h > 0.1:
            self.drone_orient[0] = np.clip(
                self.drone_vel[0] / self.max_speed * 0.3, -0.5, 0.5
            )  # roll
            self.drone_orient[1] = np.clip(
                self.drone_vel[2] / self.max_speed * 0.3, -0.5, 0.5
            )  # pitch
            self.drone_orient[2] = np.arctan2(
                self.drone_vel[2], self.drone_vel[0]
            )  # yaw

        # ── 2. 상태 정보 계산 ─────────────────────────────────────────────────
        target_dist = np.linalg.norm(
            self.drone_pos[[0, 2]] - self.target_pos
        )
        lateral_offset = np.abs(
            np.cross(
                self.drone_vel[[0, 2]] / (np.linalg.norm(self.drone_vel[[0, 2]]) + 1e-8),
                self.target_pos - self.drone_pos[[0, 2]],
            )
        )

        # 투하 타이밍 신호
        altitude = self.drone_pos[1]
        if altitude > 0:
            t_fall = (2.0 * altitude / self.gravity) ** 0.5
            drift = self.drone_vel[[0, 2]] * t_fall + 0.5 * self.wind_vel * t_fall ** 2
            expected_landing = self.drone_pos[[0, 2]] + drift
            timing_signal = (
                np.linalg.norm(expected_landing - self.target_pos) / self.world_size
            )
        else:
            timing_signal = 1.0

        speed = np.linalg.norm(self.drone_vel)

        # ── 3. 투하 처리 + 보상 ──────────────────────────────────────────────
        just_dropped = False
        landing_distance = 0.0

        if action[3] >= 0.0 and not self.has_dropped:
            self.has_dropped = True
            just_dropped = True
            landing_pos = self._compute_3d_landing()
            landing_distance = np.linalg.norm(landing_pos - self.target_pos)

        # OOB 체크
        oob = (
            abs(self.drone_pos[0]) > self.world_size
            or abs(self.drone_pos[2]) > self.world_size
        )

        timeout = self.current_step >= self.max_steps

        state = {
            "altitude": altitude,
            "speed": speed,
            "target_dist": target_dist,
            "lateral_offset": lateral_offset,
            "timing_signal": timing_signal,
        }

        info_for_reward = {
            "has_dropped": self.has_dropped and not just_dropped,
            "just_dropped": just_dropped,
            "landing_distance": landing_distance,
            "oob": oob,
            "timeout": timeout,
        }

        reward, terminated, truncated = self.reward_fn.compute(
            state, action, info_for_reward
        )

        info = {
            **self._get_info(),
            "landing_distance": landing_distance if just_dropped else None,
            "target_dist": target_dist,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_info(self) -> dict:
        return {
            "drone_pos": self.drone_pos.copy(),
            "target_pos": self.target_pos.copy(),
            "wind_vel": self.wind_vel.copy(),
            "has_dropped": self.has_dropped,
            "step": self.current_step,
        }

    # ════════════════════════════════════════════════════════════════════════
    # Pygame 렌더링 (top-down view)
    # ════════════════════════════════════════════════════════════════════════

    def render(self):
        """현재 상태를 Pygame으로 렌더링 (탑다운 뷰)."""
        if self.render_mode is None:
            return None

        try:
            import pygame
        except ImportError:
            return None

        SCREEN_SIZE = 600
        SCALE = SCREEN_SIZE / (self.world_size * 2)
        OFFSET = SCREEN_SIZE // 2

        def world_to_screen(x, z):
            return int(x * SCALE + OFFSET), int(z * SCALE + OFFSET)

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
                pygame.display.set_caption("DroneDropEnv3D — Top-Down View")
            else:
                self.screen = pygame.Surface((SCREEN_SIZE, SCREEN_SIZE))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("monospace", 14)

        # 배경
        self.screen.fill((34, 80, 34))

        # 그리드
        for i in range(-int(self.world_size), int(self.world_size) + 1, 20):
            sx, _ = world_to_screen(i, 0)
            _, sy = world_to_screen(0, i)
            pygame.draw.line(
                self.screen, (44, 90, 44), (sx, 0), (sx, SCREEN_SIZE), 1
            )
            pygame.draw.line(
                self.screen, (44, 90, 44), (0, sy), (SCREEN_SIZE, sy), 1
            )

        # 타겟
        tx, tz = world_to_screen(self.target_pos[0], self.target_pos[1])
        pygame.draw.circle(self.screen, (255, 50, 50), (tx, tz), 8, 2)
        pygame.draw.circle(self.screen, (255, 100, 100), (tx, tz), 3)

        # 드론
        dx, dz = world_to_screen(self.drone_pos[0], self.drone_pos[2])
        alt_color = min(255, int(self.drone_pos[1] / self.max_altitude * 200) + 55)
        pygame.draw.circle(
            self.screen, (50, 50, alt_color), (dx, dz), 6
        )
        # 속도 벡터
        vx_screen = self.drone_vel[0] * SCALE * 2
        vz_screen = self.drone_vel[2] * SCALE * 2
        pygame.draw.line(
            self.screen, (100, 100, 255),
            (dx, dz), (dx + int(vx_screen), dz + int(vz_screen)), 2,
        )

        # 바람 표시
        wx, wz = world_to_screen(0, 0)
        wind_sx = self.wind_vel[0] * SCALE * 5
        wind_sz = self.wind_vel[1] * SCALE * 5
        pygame.draw.line(
            self.screen, (200, 200, 100),
            (wx, wz), (wx + int(wind_sx), wz + int(wind_sz)), 2,
        )

        # HUD
        texts = [
            f"Step: {self.current_step}/{self.max_steps}",
            f"Alt: {self.drone_pos[1]:.1f}m",
            f"Speed: {np.linalg.norm(self.drone_vel):.1f}m/s",
            f"Wind: {np.linalg.norm(self.wind_vel):.1f}m/s",
            f"Dist: {np.linalg.norm(self.drone_pos[[0, 2]] - self.target_pos):.1f}m",
            f"Drop: {'Yes' if self.has_dropped else 'No'}",
        ]
        for i, text in enumerate(texts):
            surf = self.font.render(text, True, (220, 220, 220))
            self.screen.blit(surf, (8, 8 + i * 18))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        return None

    def close(self):
        if self.screen is not None:
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass
            self.screen = None
            self.clock = None
            self.font = None
