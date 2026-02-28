"""
train_ppo.py — PPO LunarLander 학습

실행:
  cd lunar_lander
  python -m train.train_ppo                # 이산 (LunarLander-v3)
  python -m train.train_ppo --continuous   # 연속 (LunarLanderContinuous-v3)
"""

import os
import sys
import time
import argparse
from collections import deque

import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from ..configs.ppo_discrete import CONFIG as DISCRETE_CONFIG
from ..configs.ppo_continuous import CONFIG as CONTINUOUS_CONFIG
from ..agents.ppo_agent import PPOAgent
from ..buffers.rollout_buffer import RolloutBuffer


def save_training_plot(ep_rewards: list, save_path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    episodes = np.arange(1, len(ep_rewards) + 1)
    ax.plot(episodes, ep_rewards, alpha=0.3, color="steelblue", label="Episode Reward")
    if len(ep_rewards) >= 20:
        window = 20
        moving_avg = np.convolve(ep_rewards, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window, len(ep_rewards) + 1), moving_avg,
                color="steelblue", linewidth=2, label=f"Moving Avg ({window} ep)")
    ax.axhline(y=200, color="green", linestyle="--", alpha=0.7, label="Solved (200)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def train(continuous: bool = False) -> None:
    config = CONTINUOUS_CONFIG if continuous else DISCRETE_CONFIG
    algo_name = config["algo_name"]

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    # 디렉토리 생성
    models_dir = config["models_dir"]
    plots_dir = config["plots_dir"]
    tb_dir = os.path.join(config["tb_dir"], algo_name)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=tb_dir)

    print(f"\n{'='*60}")
    print(f"  PPO {'Continuous' if continuous else 'Discrete'} — {config['env_id']}")
    print(f"  디바이스: {device} | 총 스텝: {config['total_timesteps']:,}")
    print(f"{'='*60}\n")

    env = gym.make(config["env_id"])
    env.reset(seed=config["seed"])

    is_continuous = config["action_type"] == "continuous"
    agent = PPOAgent(
        obs_dim=config["obs_dim"],
        action_dim=config["action_dim"],
        hidden_dim=config["hidden_dim"],
        lr=config["lr"],
        clip_eps=config["clip_eps"],
        value_coef=config["value_coef"],
        entropy_coef=config["entropy_coef"],
        max_grad_norm=config["max_grad_norm"],
        continuous=is_continuous,
        device=device,
    )

    buffer = RolloutBuffer(
        rollout_length=config["rollout_length"],
        obs_dim=config["obs_dim"],
        action_dim=config["action_dim"] if is_continuous else 1,
        continuous=is_continuous,
        gamma=config["gamma"],
        lam=config["lam"],
        device=device,
    )

    total_timesteps = config["total_timesteps"]
    rollout_length = config["rollout_length"]

    obs, _ = env.reset()
    ep_reward = 0.0
    ep_rewards_history = []
    recent_rewards = deque(maxlen=config["reward_window"])

    timestep = 0
    update_count = 0
    start_time = time.time()

    while timestep < total_timesteps:
        buffer.reset()

        for _ in range(rollout_length):
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action if not is_continuous else np.array(action) if isinstance(action, (list, tuple)) else action)
            done = terminated or truncated

            buffer.store(obs, action, log_prob, reward, value, done)
            obs = next_obs
            ep_reward += reward
            timestep += 1

            if done:
                ep_rewards_history.append(ep_reward)
                recent_rewards.append(ep_reward)
                writer.add_scalar("reward/episode", ep_reward, len(ep_rewards_history))
                ep_reward = 0.0
                obs, _ = env.reset()

        last_value = agent.get_value(obs)
        buffer.compute_gae(last_value, done)

        losses = agent.update(buffer, config["n_epochs"], config["batch_size"])
        update_count += 1

        writer.add_scalar("loss/actor", losses["actor_loss"], timestep)
        writer.add_scalar("loss/critic", losses["critic_loss"], timestep)
        writer.add_scalar("loss/entropy", losses["entropy"], timestep)
        writer.add_scalar("loss/approx_kl", losses["approx_kl"], timestep)

        if update_count % config["log_interval"] == 0 and len(recent_rewards) > 0:
            elapsed = time.time() - start_time
            progress = timestep / total_timesteps * 100
            mean_reward = np.mean(recent_rewards)

            print(f"[{timestep:>7,}/{total_timesteps:,}] {progress:5.1f}% | "
                  f"평균 보상: {mean_reward:7.1f} | "
                  f"Actor: {losses['actor_loss']:.4f} | Critic: {losses['critic_loss']:.4f} | "
                  f"KL: {losses['approx_kl']:.4f} | {elapsed:.0f}s")

            writer.add_scalar("reward/mean", mean_reward, timestep)

            if mean_reward >= config["solved_threshold"] and len(recent_rewards) == config["reward_window"]:
                print(f"\n  >>> 목표 달성! 평균 보상 {mean_reward:.1f} >= {config['solved_threshold']}")

    env.close()
    writer.close()

    print(f"\n{'='*60}")
    print(f"  학습 완료! 총 에피소드: {len(ep_rewards_history)}")
    if ep_rewards_history:
        print(f"  최종 평균 보상: {np.mean(ep_rewards_history[-20:]):.1f}")
    print(f"{'='*60}\n")

    model_path = os.path.join(models_dir, f"{algo_name}.pt")
    agent.save(model_path)

    if ep_rewards_history:
        plot_path = os.path.join(plots_dir, f"{algo_name}.png")
        save_training_plot(ep_rewards_history, plot_path, f"PPO {'Continuous' if continuous else 'Discrete'} — {config['env_id']}")
        print(f"학습 곡선 저장: {plot_path}")
        np.save(os.path.join(plots_dir, f"{algo_name}_rewards.npy"), np.array(ep_rewards_history))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuous", action="store_true", help="LunarLanderContinuous-v3 학습")
    args = parser.parse_args()
    train(continuous=args.continuous)
