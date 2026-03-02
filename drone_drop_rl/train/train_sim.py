"""
train_sim.py — SAC 드론 투하 시뮬레이션 학습

Shaped vs Sparse 보상 비교 실험 지원.

실행:
  cd "RL Workspace"
  python -m drone_drop_rl.train.train_sim                    # Shaped (기본)
  python -m drone_drop_rl.train.train_sim --reward sparse    # Sparse (대조)
  python -m drone_drop_rl.train.train_sim --curriculum       # 커리큘럼 학습
"""

import os
import time
import argparse
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from ..configs.sac_drone import CONFIG, SPARSE_CONFIG
from ..envs.drone_sim_3d import DroneDropEnv3D
from ..agents.sac_agent import SACAgent, ReplayBuffer
from .curriculum import CurriculumManager


def save_training_plot(ep_rewards: list, save_path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    episodes = np.arange(1, len(ep_rewards) + 1)
    ax.plot(episodes, ep_rewards, alpha=0.3, color="darkorange", label="Episode Reward")
    if len(ep_rewards) >= 50:
        window = 50
        moving_avg = np.convolve(ep_rewards, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window, len(ep_rewards) + 1), moving_avg,
                color="darkorange", linewidth=2, label=f"Moving Avg ({window} ep)")
    ax.axhline(y=15, color="green", linestyle="--", alpha=0.7, label="Target (15)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def train(reward_type: str = "shaped", use_curriculum: bool = False,
          total_steps_override: int = None) -> None:
    config = SPARSE_CONFIG if reward_type == "sparse" else CONFIG
    algo_name = config["algo_name"]

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    models_dir = config["models_dir"]
    plots_dir = config["plots_dir"]
    tb_dir = os.path.join(config["tb_dir"], algo_name)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=tb_dir)

    # 커리큘럼 매니저
    curriculum = CurriculumManager() if use_curriculum else None

    # 환경 생성
    env_kwargs = {
        "reward_type": reward_type,
        "world_size": config["world_size"] if "world_size" in config else 200.0,
        "max_altitude": config["max_altitude"],
        "default_altitude": config["default_altitude"],
        "gravity": config["gravity"],
        "dt": config["dt"],
        "max_steps": config["max_steps"],
        "max_speed": config["max_speed"],
        "max_wind": config["max_wind"],
        "gamma": config["gamma"],
    }
    if curriculum:
        env_kwargs.update(curriculum.get_env_kwargs())

    env = DroneDropEnv3D(**env_kwargs)
    env.reset(seed=config["seed"])

    print(f"\n{'='*60}")
    print(f"  SAC Drone Drop — {reward_type.upper()} Reward")
    if curriculum:
        print(f"  커리큘럼: {curriculum.stage_name}")
    print(f"  디바이스: {device} | 총 스텝: {config['total_timesteps']:,}")
    print(f"{'='*60}\n")

    agent = SACAgent(
        obs_dim=config["obs_dim"],
        action_dim=config["action_dim"],
        hidden_dim=config["hidden_dim"],
        lr_actor=config["lr_actor"],
        lr_critic=config["lr_critic"],
        lr_alpha=config["lr_alpha"],
        gamma=config["gamma"],
        tau=config["tau"],
        init_alpha=config["init_alpha"],
        target_entropy=config["target_entropy"],
        device=device,
    )

    buffer = ReplayBuffer(
        capacity=config["buffer_capacity"],
        obs_dim=config["obs_dim"],
        action_dim=config["action_dim"],
        device=device,
    )

    total_timesteps = total_steps_override or config["total_timesteps"]
    learning_starts = config["learning_starts"]
    train_frequency = config["train_frequency"]
    batch_size = config["batch_size"]
    reward_window = config["reward_window"]

    ep_rewards_history = []
    landing_distances = []
    recent_rewards = deque(maxlen=reward_window)

    obs, _ = env.reset()
    ep_reward = 0.0
    timestep = 0
    start_time = time.time()

    while timestep < total_timesteps:
        if timestep < learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        buffer.store(obs, action, reward, next_obs, done)
        obs = next_obs
        ep_reward += reward
        timestep += 1

        # 학습
        if timestep >= learning_starts and timestep % train_frequency == 0:
            losses = agent.update(buffer, batch_size)

            if timestep % 2000 == 0:
                writer.add_scalar("loss/critic", losses["critic_loss"], timestep)
                writer.add_scalar("loss/actor", losses["actor_loss"], timestep)
                writer.add_scalar("loss/alpha", losses["alpha_loss"], timestep)
                writer.add_scalar("train/alpha", losses["alpha"], timestep)
                writer.add_scalar("train/q_mean", losses["q_mean"], timestep)

        if done:
            ep_rewards_history.append(ep_reward)
            recent_rewards.append(ep_reward)
            writer.add_scalar("reward/episode", ep_reward, len(ep_rewards_history))

            # 착지 거리 기록
            landing_dist = info.get("landing_distance")
            if landing_dist is not None:
                landing_distances.append(landing_dist)
                writer.add_scalar("env/landing_distance", landing_dist, len(ep_rewards_history))

            # 커리큘럼 승급 체크
            if curriculum:
                promoted = curriculum.record_episode(landing_dist)
                if promoted:
                    print(f"\n  >>> 커리큘럼 승급! → {curriculum.stage_name}")
                    # 새 단계 환경 재생성
                    env.close()
                    env_kwargs.update(curriculum.get_env_kwargs())
                    env = DroneDropEnv3D(**env_kwargs)

                if len(ep_rewards_history) % 50 == 0:
                    writer.add_scalar("curriculum/stage", curriculum.current_stage_idx, timestep)
                    writer.add_scalar("curriculum/success_rate", curriculum.success_rate, timestep)

            # 로깅
            if len(ep_rewards_history) % config["log_interval"] == 0 and len(recent_rewards) > 0:
                elapsed = time.time() - start_time
                progress = timestep / total_timesteps * 100
                mean_reward = np.mean(recent_rewards)

                stage_info = f" | {curriculum.summary()}" if curriculum else ""
                print(
                    f"[{timestep:>8,}/{total_timesteps:,}] {progress:5.1f}% | "
                    f"에피소드: {len(ep_rewards_history)} | "
                    f"평균 보상: {mean_reward:7.2f} | "
                    f"alpha: {agent.alpha.item():.3f}"
                    f"{stage_info} | {elapsed:.0f}s"
                )

                writer.add_scalar("reward/mean", mean_reward, timestep)

            ep_reward = 0.0
            obs, _ = env.reset()

    env.close()
    writer.close()

    # ── 결과 저장 ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  학습 완료! 총 에피소드: {len(ep_rewards_history)}")
    if ep_rewards_history:
        print(f"  최종 평균 보상 (last {reward_window}): {np.mean(list(recent_rewards)):.2f}")
    if landing_distances:
        recent_dists = landing_distances[-100:]
        print(f"  최종 평균 착지 거리 (last 100): {np.mean(recent_dists):.2f}m")
        success_count = sum(1 for d in recent_dists if d < 5.0)
        print(f"  성공률 (<5m): {success_count / len(recent_dists):.1%}")
    print(f"{'='*60}\n")

    model_path = os.path.join(models_dir, f"{algo_name}.pt")
    agent.save(model_path)

    if ep_rewards_history:
        plot_path = os.path.join(plots_dir, f"{algo_name}.png")
        title = f"SAC Drone Drop — {reward_type.capitalize()} Reward"
        save_training_plot(ep_rewards_history, plot_path, title)
        print(f"학습 곡선 저장: {plot_path}")
        np.save(os.path.join(plots_dir, f"{algo_name}_rewards.npy"), np.array(ep_rewards_history))
        if landing_distances:
            np.save(os.path.join(plots_dir, f"{algo_name}_distances.npy"), np.array(landing_distances))


def main():
    parser = argparse.ArgumentParser(description="SAC Drone Drop Training")
    parser.add_argument("--reward", type=str, default="shaped",
                        choices=["shaped", "sparse"],
                        help="보상 함수 타입 (shaped 또는 sparse)")
    parser.add_argument("--curriculum", action="store_true",
                        help="커리큘럼 학습 사용")
    parser.add_argument("--steps", type=int, default=None,
                        help="총 학습 스텝 수 (기본: config 값)")
    args = parser.parse_args()

    train(reward_type=args.reward, use_curriculum=args.curriculum,
          total_steps_override=args.steps)


if __name__ == "__main__":
    main()
