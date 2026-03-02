"""
evaluate.py — 학습된 모델 평가 + 렌더링

실행:
  cd "RL Workspace"
  python -m drone_drop_rl.evaluate --algo sac_shaped --render
  python -m drone_drop_rl.evaluate --algo sac_sparse --episodes 50
"""

import argparse
import numpy as np
import torch

from .configs.sac_drone import CONFIG, SPARSE_CONFIG
from .envs.drone_sim_3d import DroneDropEnv3D
from .agents.sac_agent import SACAgent


def evaluate(algo_name: str, num_episodes: int = 50, render: bool = False):
    config = SPARSE_CONFIG if "sparse" in algo_name else CONFIG
    device = torch.device("cpu")

    reward_type = "sparse" if "sparse" in algo_name else "shaped"
    render_mode = "human" if render else None

    env = DroneDropEnv3D(
        render_mode=render_mode,
        reward_type=reward_type,
        max_wind=config["max_wind"],
    )

    agent = SACAgent(
        obs_dim=config["obs_dim"],
        action_dim=config["action_dim"],
        hidden_dim=config["hidden_dim"],
        device=device,
    )

    import os
    model_path = os.path.join(config["models_dir"], f"{algo_name}.pt")
    agent.load(model_path)

    print(f"\n{'='*60}")
    print(f"  평가: {algo_name} | {num_episodes} 에피소드")
    print(f"{'='*60}\n")

    rewards = []
    distances = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.deterministic_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

            if render:
                env.render()

        rewards.append(ep_reward)
        landing_dist = info.get("landing_distance")
        if landing_dist is not None:
            distances.append(landing_dist)

        status = f"거리: {landing_dist:.2f}m" if landing_dist is not None else "미투하"
        print(f"  에피소드 {ep + 1:>3d} | 보상: {ep_reward:>7.2f} | {status}")

    env.close()

    print(f"\n{'='*60}")
    print(f"  결과 요약")
    print(f"  평균 보상: {np.mean(rewards):.2f} (±{np.std(rewards):.2f})")
    if distances:
        print(f"  평균 착지 거리: {np.mean(distances):.2f}m (±{np.std(distances):.2f}m)")
        success = sum(1 for d in distances if d < 5.0)
        precision = sum(1 for d in distances if d < 1.0)
        print(f"  성공률 (<5m): {success / len(distances):.1%}")
        print(f"  정밀도 (<1m): {precision / len(distances):.1%}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained drone drop agent")
    parser.add_argument("--algo", type=str, default="sac_shaped",
                        choices=["sac_shaped", "sac_sparse"],
                        help="평가할 알고리즘")
    parser.add_argument("--episodes", type=int, default=50,
                        help="평가 에피소드 수")
    parser.add_argument("--render", action="store_true",
                        help="Pygame 렌더링")
    args = parser.parse_args()

    evaluate(args.algo, args.episodes, args.render)


if __name__ == "__main__":
    main()
