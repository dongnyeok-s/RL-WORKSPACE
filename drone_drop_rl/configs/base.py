"""공유 설정 — 드론 투하 RL 시스템 기본 CONFIG."""

import os as _os

_PACKAGE_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_RESULTS = _os.path.join(_PACKAGE_DIR, "results")

BASE_CONFIG = {
    # ── 환경 ─────────────────────────────────────────────────────────────────
    "obs_dim": 14,
    "action_dim": 4,

    # ── 물리 (시뮬레이션) ──────────────────────────────────────────────────────
    "world_size": 80.0,         # 월드 크기 (m) — 학습 효율을 위해 축소
    "max_altitude": 60.0,       # 최대 고도 (m)
    "default_altitude": 40.0,   # 기본 비행 고도 (m)
    "gravity": 9.81,            # 중력 가속도 (m/s²)
    "dt": 1.0 / 30.0,           # 시뮬레이션 타임스텝 (30Hz)
    "max_steps": 600,           # 최대 스텝 수 (20초)
    "max_speed": 20.0,          # 최대 드론 속도 (m/s)
    "max_wind": 5.0,            # 최대 풍속 (m/s)

    # ── 학습 ─────────────────────────────────────────────────────────────────
    "total_timesteps": 2_000_000,
    "seed": 42,

    # ── 로깅 ─────────────────────────────────────────────────────────────────
    "reward_window": 50,
    "log_interval": 10,
    "results_dir": _RESULTS,
    "tb_dir": _os.path.join(_RESULTS, "tensorboard"),
    "models_dir": _os.path.join(_RESULTS, "models"),
    "plots_dir": _os.path.join(_RESULTS, "plots"),

    # ── 평가 ─────────────────────────────────────────────────────────────────
    "eval_episodes": 50,
    "solved_threshold": 15.0,   # 정밀 투하 평균 보상
}
