"""DQN (LunarLander-v3) 하이퍼파라미터."""

from .base import BASE_CONFIG

CONFIG = {
    **BASE_CONFIG,
    "algo_name": "dqn",
    "env_id": BASE_CONFIG["env_discrete"],
    "action_dim": BASE_CONFIG["discrete_action_dim"],
    "action_type": "discrete",

    "buffer_capacity": 100_000,
    "batch_size": 64,
    "lr": 1e-3,
    "gamma": 0.99,
    "tau": 0.005,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay": 50_000,
    "learning_starts": 10_000,
    "train_frequency": 4,
    "hidden_dim": 64,
}
