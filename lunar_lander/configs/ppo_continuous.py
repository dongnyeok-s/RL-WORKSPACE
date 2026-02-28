"""PPO 연속 (LunarLanderContinuous-v3) 하이퍼파라미터."""

from .base import BASE_CONFIG

CONFIG = {
    **BASE_CONFIG,
    "algo_name": "ppo_continuous",
    "env_id": BASE_CONFIG["env_continuous"],
    "action_dim": BASE_CONFIG["continuous_action_dim"],
    "action_type": "continuous",

    "rollout_length": 2048,
    "n_epochs": 10,
    "batch_size": 64,
    "lr": 3e-4,
    "gamma": 0.99,
    "lam": 0.95,
    "clip_eps": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.0,
    "max_grad_norm": 0.5,
    "hidden_dim": 64,
}
