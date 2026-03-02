"""SAC 드론 투하 하이퍼파라미터."""

from .base import BASE_CONFIG

CONFIG = {
    **BASE_CONFIG,
    "algo_name": "sac_shaped",
    "action_type": "continuous",

    "buffer_capacity": 500_000,
    "batch_size": 256,
    "lr_actor": 1e-4,
    "lr_critic": 3e-4,
    "lr_alpha": 1e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "init_alpha": 0.2,
    "target_entropy": -4.0,     # -dim(action)
    "learning_starts": 25_000,
    "train_frequency": 1,
    "hidden_dim": 256,

    # 보상 모드
    "reward_type": "shaped",    # "shaped" 또는 "sparse"
}

# Sparse 대조 실험 설정
SPARSE_CONFIG = {
    **CONFIG,
    "algo_name": "sac_sparse",
    "reward_type": "sparse",
}
