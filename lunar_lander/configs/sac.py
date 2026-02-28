"""SAC (LunarLanderContinuous-v3) 하이퍼파라미터."""

from .base import BASE_CONFIG

CONFIG = {
    **BASE_CONFIG,
    "algo_name": "sac",
    "env_id": BASE_CONFIG["env_continuous"],
    "action_dim": BASE_CONFIG["continuous_action_dim"],
    "action_type": "continuous",

    "buffer_capacity": 100_000,
    "batch_size": 256,
    "lr_actor": 3e-4,
    "lr_critic": 3e-4,
    "lr_alpha": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "init_alpha": 0.2,
    "target_entropy": -2.0,
    "learning_starts": 10_000,
    "train_frequency": 1,
    "hidden_dim": 256,
}
