"""공유 설정 — 모든 알고리즘이 참조하는 기본 CONFIG."""

import os as _os

# 경로를 lunar_lander/ 디렉토리 기준으로 설정
_PACKAGE_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_RESULTS = _os.path.join(_PACKAGE_DIR, "results")

BASE_CONFIG = {
    # ── 환경 ─────────────────────────────────────────────────────────────────
    "env_discrete": "LunarLander-v3",
    "env_continuous": "LunarLanderContinuous-v3",
    "obs_dim": 8,
    "discrete_action_dim": 4,
    "continuous_action_dim": 2,

    # ── 학습 ─────────────────────────────────────────────────────────────────
    "total_timesteps": 500_000,
    "seed": 42,

    # ── 로깅 ─────────────────────────────────────────────────────────────────
    "reward_window": 20,
    "log_interval": 1,
    "results_dir": _RESULTS,
    "tb_dir": _os.path.join(_RESULTS, "tensorboard"),
    "models_dir": _os.path.join(_RESULTS, "models"),
    "plots_dir": _os.path.join(_RESULTS, "plots"),

    # ── 평가 ─────────────────────────────────────────────────────────────────
    "eval_episodes": 20,
    "solved_threshold": 200.0,
}
