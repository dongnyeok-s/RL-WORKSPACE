"""
train_reinforce.py — REINFORCE LunarLander 학습

실행:
  cd lunar_lander
  python -m train.train_reinforce
"""

import os
import time
from collections import deque

import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from ..configs.reinforce import CONFIG
from ..agents.reinforce_agent import REINFORCEAgent


def save_training_plot(ep_rewards: list, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    episodes = np.arange(1, len(ep_rewards) + 1)
    ax.plot(episodes, ep_rewards, alpha=0.3, color="coral", label="Episode Reward")
    if len(ep_rewards) >= 20:
        window = 20
        moving_avg = np.convolve(ep_rewards, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window, len(ep_rewards) + 1), moving_avg,
                color="coral", linewidth=2, label=f"Moving Avg ({window} ep)")
    ax.axhline(y=200, color="green", linestyle="--", alpha=0.7, label="Solved (200)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("REINFORCE — LunarLander-v3")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def train() -> None:
    config = CONFIG
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

    print(f"\n{'='*60}")
    print(f"  REINFORCE — {config['env_id']}")
    print(f"  디바이스: {device} | 총 스텝: {config['total_timesteps']:,}")
    print(f"{'='*60}\n")

    env = gym.make(config["env_id"])
    env.reset(seed=config["seed"])

    agent = REINFORCEAgent(
        obs_dim=config["obs_dim"],
        action_dim=config["action_dim"],
        hidden_dim=config["hidden_dim"],
        lr=config["lr"],
        gamma=config["gamma"],
        device=device,
    )

    total_timesteps = config["total_timesteps"]
    ep_rewards_history = []
    recent_rewards = deque(maxlen=config["reward_window"])

    timestep = 0
    start_time = time.time()

    while timestep < total_timesteps:
        obs, _ = env.reset()
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_reward(reward)
            obs = next_obs
            timestep += 1

        result = agent.update()
        ep_rewards_history.append(result["ep_reward"])
        recent_rewards.append(result["ep_reward"])

        writer.add_scalar("reward/episode", result["ep_reward"], len(ep_rewards_history))
        writer.add_scalar("loss/policy", result["loss"], len(ep_rewards_history))

        if len(ep_rewards_history) % 10 == 0 and len(recent_rewards) > 0:
            elapsed = time.time() - start_time
            progress = timestep / total_timesteps * 100
            mean_reward = np.mean(recent_rewards)

            print(f"[{timestep:>7,}/{total_timesteps:,}] {progress:5.1f}% | "
                  f"에피소드: {len(ep_rewards_history)} | "
                  f"평균 보상: {mean_reward:7.1f} | "
                  f"Loss: {result['loss']:.4f} | {elapsed:.0f}s")

            writer.add_scalar("reward/mean", mean_reward, timestep)

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
        save_training_plot(ep_rewards_history, plot_path)
        print(f"학습 곡선 저장: {plot_path}")
        np.save(os.path.join(plots_dir, f"{algo_name}_rewards.npy"), np.array(ep_rewards_history))


if __name__ == "__main__":
    train()
