"""REINFORCE (LunarLander-v3) 하이퍼파라미터."""

from .base import BASE_CONFIG

CONFIG = {
    **BASE_CONFIG,
    "algo_name": "reinforce",
    "env_id": BASE_CONFIG["env_discrete"],
    "action_dim": BASE_CONFIG["discrete_action_dim"],
    "action_type": "discrete",

    "lr": 1e-3,
    "gamma": 0.99,
    "hidden_dim": 64,
}
