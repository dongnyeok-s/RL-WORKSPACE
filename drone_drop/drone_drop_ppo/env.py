"""
env.py - 2D 드론 투하 커스텀 Gymnasium 환경

[환경 설명]
드론이 고정 고도에서 왼쪽→오른쪽으로 수평 비행한다.
목표 지점은 고정이고, 드론 속도가 매 에피소드마다 달라진다.
에이전트는 현재 속도를 관측하고 적절한 타이밍에 패키지를 투하하여
지상의 목표 지점에 최대한 가깝게 착지시키는 것이 목표이다.

[핵심 물리]
  낙하 시간: t_fall = sqrt(2 * ALTITUDE / GRAVITY) ≈ 2.26s (고정)
  수평 이동: drift = speed * t_fall (속도에 비례)
  최적 투하 지점: drone_x = target_x - drift

  → 에이전트는 "속도에 따른 최적 투하 타이밍"을 일반화해서 학습해야 한다.

[관측 공간] (4차원, 정규화)
  0: drone_x / WORLD_WIDTH                    드론 수평 위치 [0, ~1.5]
  1: drone_speed / MAX_SPEED                  현재 속도 (정규화) [0.375, 1]
  2: (drone_x - target_x) / WORLD_WIDTH       상대 거리 [-1, 1.5]
  3: (drift - remaining_dist) / WORLD_WIDTH   투하 타이밍 신호
                                              0에 가까울수록 지금이 최적 투하 시점

[행동 공간] (이산, 2개)
  0: WAIT  - 대기
  1: DROP  - 패키지 투하 (1회만 가능)

[보상]
  - 매 스텝: -0.01 (시간 페널티)
  - 착지 시: max(0, 1 - dist/400) * 10 + (dist < 20 이면 +5 보너스)
  - 미투하 화면 이탈: -5.0

[렌더링]
  Pygame으로 드론, 목표, 패키지 궤적을 시각화.
  render_mode="human" (실시간 창) / "rgb_array" (프레임 캡처)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ════════════════════════════════════════════════════════════════════════════
# 물리 상수
# ════════════════════════════════════════════════════════════════════════════
WORLD_WIDTH = 800.0       # 월드 너비 (px)
WORLD_HEIGHT = 600.0      # 월드 높이 (px)
DRONE_ALTITUDE = 500.0    # 드론 고정 고도 (px, 지면 기준)
TARGET_X = 400.0          # 목표 지점 고정 (화면 중앙)
MIN_SPEED = 60.0          # 드론 최소 속도 (px/s)
MAX_SPEED = 160.0         # 드론 최대 속도 (px/s)
                          # 최대 drift = 160 * 2.26 ≈ 361px → 최적 투하 x ≈ 39px (항상 양수)
GRAVITY = 9.81 * 20       # 중력 가속도 (시각 효과를 위해 스케일업)
DT = 1.0 / 30.0           # 시뮬레이션 타임스텝 (30Hz)
MAX_STEPS = 300           # 최대 스텝 수 (10초)
GROUND_HEIGHT = 20        # 지면 렌더링 높이 (px)

# 낙하 시간 (고정): t = sqrt(2h/g) ≈ 2.26s
FALL_TIME = (2.0 * DRONE_ALTITUDE / GRAVITY) ** 0.5


class DroneDropEnv(gym.Env):
    """
    2D 드론 투하 환경 (고정 목표 + 가변 속도).

    매 에피소드마다 드론 속도가 [60, 200] px/s 범위에서 랜덤 결정된다.
    에이전트는 속도를 관측하고, 적절한 타이밍에 투하해야 한다.

    Args:
        render_mode: "human" (pygame 창) 또는 "rgb_array" (프레임 반환)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        # ── 관측/행동 공간 정의 ─────────────────────────────────────────────
        # [drone_x/W, speed/MAX_SPEED, (drone_x-target_x)/W, timing_signal]
        self.observation_space = spaces.Box(
            low=np.array([-0.5, 0.0, -1.5, -1.5], dtype=np.float32),
            high=np.array([1.5, 1.0, 1.5, 1.5], dtype=np.float32),
        )
        self.action_space = spaces.Discrete(2)  # 0=WAIT, 1=DROP

        # ── 상태 변수 (reset에서 초기화) ────────────────────────────────────
        self.drone_x = 0.0
        self.drone_speed = 0.0
        self.target_x = TARGET_X
        self.has_dropped = False
        self.pkg_landed = False
        self.pkg_x = 0.0
        self.pkg_y = 0.0
        self.pkg_vx = 0.0
        self.pkg_vy = 0.0
        self.current_step = 0
        self.drop_x = None  # 투하 시점의 드론 x좌표 (로깅용)

        # ── Pygame 렌더링 ───────────────────────────────────────────────────
        self.screen = None
        self.clock = None
        self.font = None

    def reset(self, seed=None, options=None):
        """환경을 초기화하고 첫 관측을 반환한다."""
        super().reset(seed=seed)

        self.drone_x = 0.0
        self.drone_speed = self.np_random.uniform(MIN_SPEED, MAX_SPEED)
        self.target_x = TARGET_X
        self.has_dropped = False
        self.pkg_landed = False
        self.pkg_x = 0.0
        self.pkg_y = DRONE_ALTITUDE
        self.pkg_vx = 0.0
        self.pkg_vy = 0.0
        self.current_step = 0
        self.drop_x = None

        # 최적 투하 지점 계산 (참고용, 에이전트에게 직접 제공하지 않음)
        drift = self.drone_speed * FALL_TIME
        self._optimal_drop_x = self.target_x - drift

        return self._get_obs(), {"speed": self.drone_speed}

    def _get_obs(self):
        """
        현재 상태를 정규화된 관측 벡터로 변환한다.

        obs[3]은 "투하 타이밍 신호":
          drift = speed * t_fall (패키지가 낙하하는 동안 수평 이동하는 거리)
          remaining = target_x - drone_x (목표까지 남은 거리)
          timing = (drift - remaining) / WORLD_WIDTH
            → 음수: 아직 일찍, 양수: 이미 늦음, 0 근처: 지금이 최적
        """
        drift = self.drone_speed * FALL_TIME
        remaining = self.target_x - self.drone_x
        timing_signal = (drift - remaining) / WORLD_WIDTH

        return np.array([
            self.drone_x / WORLD_WIDTH,
            self.drone_speed / MAX_SPEED,
            (self.drone_x - self.target_x) / WORLD_WIDTH,
            timing_signal,
        ], dtype=np.float32)

    def _compute_landing_x(self, drop_x: float) -> float:
        """
        투하 지점에서 패키지의 착지 x좌표를 해석적으로 계산한다.
        landing_x = drop_x + speed * t_fall  (공기 저항 없음)
        """
        return drop_x + self.drone_speed * FALL_TIME

    def step(self, action):
        """
        한 스텝 진행.

        투하(DROP) 시 착지 위치를 즉시 해석적으로 계산하고 에피소드를 종료한다.
        렌더링 모드에서는 실제 포물선 낙하를 시뮬레이션하여 시각화한다.

        Args:
            action: 0 (WAIT) 또는 1 (DROP)

        Returns:
            obs, reward, terminated, truncated, info
        """
        self.current_step += 1

        # ── 1. 드론 이동 ────────────────────────────────────────────────────
        self.drone_x += self.drone_speed * DT

        # ── 2. 투하 처리 ────────────────────────────────────────────────────
        terminated = False
        truncated = False
        distance = None
        reward = -0.01  # 시간 페널티

        if action == 1 and not self.has_dropped:
            self.has_dropped = True
            self.drop_x = self.drone_x
            self.pkg_x = self.drone_x
            self.pkg_y = DRONE_ALTITUDE
            self.pkg_vx = self.drone_speed
            self.pkg_vy = 0.0

            # 착지 위치를 해석적으로 즉시 계산 → 즉각적인 보상 신호
            landing_x = self._compute_landing_x(self.drone_x)
            distance = abs(landing_x - self.target_x)

            # 착지 보상: 거리에 반비례
            reward = max(0.0, 1.0 - distance / 400.0) * 10.0
            if distance < 20.0:
                reward += 5.0  # 정밀 착지 보너스

            # 렌더링용: 실제 착지 위치 기록
            self.pkg_x = landing_x
            self.pkg_y = 0.0
            self.pkg_landed = True

            terminated = True

        elif self.drone_x > WORLD_WIDTH + 200 and not self.has_dropped:
            # 투하 없이 화면 이탈
            reward = -5.0
            truncated = True

        elif self.current_step >= MAX_STEPS:
            # 최대 스텝 도달
            reward = -3.0
            truncated = True

        info = {
            "distance": distance,
            "drop_x": self.drop_x,
            "speed": self.drone_speed,
            "optimal_drop_x": self._optimal_drop_x,
        }

        return self._get_obs(), reward, terminated, truncated, info

    # ════════════════════════════════════════════════════════════════════════
    # Pygame 렌더링
    # ════════════════════════════════════════════════════════════════════════

    def render(self):
        """현재 상태를 Pygame으로 렌더링한다."""
        if self.render_mode is None:
            return None

        try:
            import pygame
        except ImportError:
            return None

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode(
                    (int(WORLD_WIDTH), int(WORLD_HEIGHT))
                )
                pygame.display.set_caption("Drone Drop - RL")
            else:
                self.screen = pygame.Surface(
                    (int(WORLD_WIDTH), int(WORLD_HEIGHT))
                )
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("monospace", 18)

        # ── 배경 (하늘) ────────────────────────────────────────────────────
        self.screen.fill((135, 206, 235))

        # ── 지면 ───────────────────────────────────────────────────────────
        pygame.draw.rect(
            self.screen, (34, 139, 34),
            (0, int(WORLD_HEIGHT) - GROUND_HEIGHT, int(WORLD_WIDTH), GROUND_HEIGHT),
        )

        # ── 목표 지점 (빨간 깃발) ──────────────────────────────────────────
        tx = int(self.target_x)
        ground_y = int(WORLD_HEIGHT) - GROUND_HEIGHT
        # 기둥
        pygame.draw.line(self.screen, (255, 0, 0), (tx, ground_y), (tx, ground_y - 40), 3)
        # 깃발 삼각형
        pygame.draw.polygon(self.screen, (255, 0, 0), [
            (tx, ground_y - 40),
            (tx + 20, ground_y - 32),
            (tx, ground_y - 24),
        ])
        # 목표 원
        pygame.draw.circle(self.screen, (255, 50, 50), (tx, ground_y - 5), 6, 2)

        # ── 최적 투하 지점 표시 (점선, 참고용) ─────────────────────────────
        optimal_x = int(self._optimal_drop_x)
        if 0 <= optimal_x <= int(WORLD_WIDTH):
            drone_y = int(WORLD_HEIGHT - DRONE_ALTITUDE)
            for y in range(drone_y, ground_y, 8):
                pygame.draw.line(
                    self.screen, (100, 200, 100),
                    (optimal_x, y), (optimal_x, min(y + 4, ground_y)), 1,
                )

        # ── 드론 ───────────────────────────────────────────────────────────
        dx = int(self.drone_x)
        dy = int(WORLD_HEIGHT - DRONE_ALTITUDE)
        # 본체
        pygame.draw.rect(self.screen, (50, 50, 50), (dx - 20, dy - 5, 40, 10))
        # 프로펠러 팔
        pygame.draw.line(self.screen, (80, 80, 80), (dx - 20, dy - 5), (dx - 30, dy - 10), 2)
        pygame.draw.line(self.screen, (80, 80, 80), (dx + 20, dy - 5), (dx + 30, dy - 10), 2)
        # 프로펠러 원
        pygame.draw.circle(self.screen, (150, 150, 150), (dx - 30, dy - 12), 8, 1)
        pygame.draw.circle(self.screen, (150, 150, 150), (dx + 30, dy - 12), 8, 1)

        # ── 패키지 ─────────────────────────────────────────────────────────
        if self.has_dropped:
            px = int(self.pkg_x)
            py = int(WORLD_HEIGHT - self.pkg_y)
            pygame.draw.rect(self.screen, (139, 69, 19), (px - 5, py - 5, 10, 10))
            # 착지 시 거리 표시선
            if self.pkg_landed:
                pygame.draw.line(
                    self.screen, (255, 255, 0),
                    (px, ground_y), (tx, ground_y), 2,
                )

        # ── HUD (텍스트 정보) ───────────────────────────────────────────────
        step_text = self.font.render(f"Step: {self.current_step}/{MAX_STEPS}", True, (0, 0, 0))
        self.screen.blit(step_text, (10, 10))

        speed_text = self.font.render(f"Speed: {self.drone_speed:.0f} px/s", True, (0, 0, 0))
        self.screen.blit(speed_text, (10, 35))

        status = "Dropped" if self.has_dropped else "Flying"
        status_text = self.font.render(f"Status: {status}", True, (0, 0, 0))
        self.screen.blit(status_text, (10, 60))

        if self.pkg_landed:
            dist = abs(self.pkg_x - self.target_x)
            dist_text = self.font.render(f"Distance: {dist:.1f}px", True, (200, 0, 0))
            self.screen.blit(dist_text, (10, 85))

        # ── 화면 갱신 ──────────────────────────────────────────────────────
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)),
                axes=(1, 0, 2),
            )
        return None

    def close(self):
        """Pygame 리소스를 정리한다."""
        if self.screen is not None:
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass
            self.screen = None
            self.clock = None
            self.font = None
