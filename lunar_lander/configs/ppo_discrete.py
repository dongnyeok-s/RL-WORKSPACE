"""PPO 이산 (LunarLander-v3) 하이퍼파라미터."""

from .base import BASE_CONFIG

CONFIG = {
    **BASE_CONFIG,
    "algo_name": "ppo_discrete",
    "env_id": BASE_CONFIG["env_discrete"],
    "action_dim": BASE_CONFIG["discrete_action_dim"],
    "action_type": "discrete",

    "rollout_length": 2048,
    "n_epochs": 10,
    "batch_size": 64,
    "lr": 3e-4,
    "gamma": 0.99,
    "lam": 0.95,
    "clip_eps": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5,
    "hidden_dim": 64,
}
