"""
train_pg.py - REINFORCE (Vanilla Policy Gradient) 드론 투하 학습 스크립트

PPO와의 비교를 위해 동일한 총 스텝 수(300K)로 학습한다.
학습 완료 후 PPO 결과가 있으면 비교 그래프를 자동 생성한다.

실행:
  cd "RL Workspace/drone_drop"
  python -m drone_drop_ppo.train_pg

[PPO와의 구조적 차이]
  PPO train.py                    REINFORCE train_pg.py
  ──────────────────────────     ──────────────────────────
  2048 스텝 rollout 수집          에피소드 1개 완주
  GAE 어드밴티지 계산             Monte Carlo G_t 계산
  10 epoch × 64 미니배치 학습    1번만 업데이트
  Clip으로 업데이트 제한          Clip 없음

[TensorBoard 실행]
  tensorboard --logdir results/tensorboard
"""

import os
import time
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from .pg_agent import REINFORCEAgent
from .env import DroneDropEnv
from .train import CONFIG  # 동일한 환경 설정과 결과 경로 사용


def save_pg_plot(ep_rewards: list, save_path: str) -> None:
    """REINFORCE 학습 곡선을 저장한다."""
    fig, ax = plt.subplots(figsize=(10, 5))

    episodes = np.arange(1, len(ep_rewards) + 1)
    ax.plot(episodes, ep_rewards, alpha=0.2, color="tomato", label="Episode Reward")

    if len(ep_rewards) >= 20:
        window = 20
        moving_avg = np.convolve(ep_rewards, np.ones(window) / window, mode="valid")
        ax.plot(
            np.arange(window, len(ep_rewards) + 1),
            moving_avg,
            color="tomato",
            linewidth=2,
            label=f"Moving Average ({window} ep)",
        )

    ax.axhline(y=12.0, color="green", linestyle="--", alpha=0.7, label="Target (12.0)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("REINFORCE (Vanilla PG) Drone Drop Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def save_comparison_plot(
    ppo_rewards: list,
    pg_rewards: list,
    save_path: str,
) -> None:
    """PPO와 REINFORCE 학습 곡선을 하나의 그래프에 비교한다."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, rewards, color, title in [
        (axes[0], pg_rewards, "tomato", "REINFORCE (Vanilla PG)"),
        (axes[1], ppo_rewards, "steelblue", "PPO"),
    ]:
        episodes = np.arange(1, len(rewards) + 1)
        ax.plot(episodes, rewards, alpha=0.2, color=color)
        if len(rewards) >= 20:
            window = 20
            avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(np.arange(window, len(rewards) + 1), avg, color=color, linewidth=2, label="Moving Avg (20)")
        ax.axhline(y=12.0, color="green", linestyle="--", alpha=0.7, label="Target (12.0)")
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("REINFORCE vs PPO — Drone Drop", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"비교 그래프 저장: {save_path}")


def train_pg(total_timesteps: int | None = None) -> list:
    """
    REINFORCE 드론 투하 학습 메인 함수.

    Returns:
        ep_rewards_history: 모든 에피소드 보상 리스트
    """
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])

    results_dir = CONFIG["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    if total_timesteps is None:
        total_timesteps = CONFIG["total_timesteps"]

    print(f"\n{'='*60}")
    print(f"  REINFORCE (Vanilla PG) Drone Drop 학습 시작")
    print(f"  총 스텝: {total_timesteps:,}")
    print(f"  [주목] 학습 곡선이 PPO보다 불안정할 것입니다.")
    print(f"{'='*60}\n")

    # ── TensorBoard (PG 전용 서브디렉토리) ──────────────────────────────────
    writer = SummaryWriter(log_dir=os.path.join(CONFIG["tb_dir"], "reinforce"))

    env = DroneDropEnv()
    env.reset(seed=CONFIG["seed"])

    agent = REINFORCEAgent(
        obs_dim=CONFIG["obs_dim"],
        action_dim=CONFIG["action_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        lr=CONFIG["lr"],
        gamma=CONFIG["gamma"],
    )

    timestep = 0
    episode_count = 0
    ep_rewards_history = []
    recent_rewards = deque(maxlen=CONFIG["reward_window"])

    # 환경 특화 지표
    landing_distances = []
    success_count = 0
    start_time = time.time()

    # ════════════════════════════════════════════════════════════════════════
    # 메인 학습 루프 (에피소드 기반)
    # ════════════════════════════════════════════════════════════════════════
    while timestep < total_timesteps:

        # ── ① 에피소드 완주 ──────────────────────────────────────────────────
        obs, _ = env.reset()
        done = False

        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store_reward(reward)
            timestep += 1

        # ── ② G_t 계산 + 1번 업데이트 ────────────────────────────────────────
        result = agent.update()

        ep_rewards_history.append(result["ep_reward"])
        recent_rewards.append(result["ep_reward"])
        episode_count += 1

        # 환경 특화 지표 수집
        distance = info.get("distance")
        if distance is not None:
            landing_distances.append(distance)
            if distance < 30.0:
                success_count += 1

        # ── ③ TensorBoard 로깅 ───────────────────────────────────────────────
        writer.add_scalar("Reward/episode_reward", result["ep_reward"], episode_count)
        writer.add_scalar("Loss/policy_loss", result["loss"], episode_count)

        if distance is not None:
            writer.add_scalar("Environment/landing_distance", distance, episode_count)
        if info.get("drop_x") is not None:
            writer.add_scalar("Environment/drop_timing", info["drop_x"], episode_count)
        if episode_count > 0:
            writer.add_scalar("Environment/success_rate", success_count / episode_count * 100, episode_count)
        if len(recent_rewards) > 0:
            writer.add_scalar("Reward/mean_reward", np.mean(recent_rewards), episode_count)

        # ── ④ 터미널 로깅 (100 에피소드마다) ────────────────────────────────
        if episode_count % 100 == 0:
            elapsed = time.time() - start_time
            progress = timestep / total_timesteps * 100
            mean_reward = np.mean(recent_rewards)

            print(f"{'─'*60}")
            print(f"  [Step {timestep:>7,} / {total_timesteps:,}]  진행률: {progress:5.1f}%")
            print(f"  에피소드 수: {episode_count}")
            print(f"  평균 보상 (최근 {CONFIG['reward_window']}개): {mean_reward:7.2f}")
            print(f"  Loss: {result['loss']:10.2f}  (PPO보다 불안정할 수 있음)")

            if landing_distances:
                recent_dist = landing_distances[-min(20, len(landing_distances)):]
                print(f"  평균 착지 거리: {np.mean(recent_dist):7.1f}px  |  성공률: {success_count/episode_count*100:.1f}%")

            print(f"  경과 시간: {elapsed:.1f}s")

    env.close()
    writer.close()

    print(f"\n{'='*60}")
    print(f"  REINFORCE 학습 완료!")
    print(f"  총 에피소드: {episode_count}")
    if ep_rewards_history:
        print(f"  최종 평균 보상 (전체): {np.mean(ep_rewards_history):.2f}")
        print(f"  최고 보상: {np.max(ep_rewards_history):.2f}")
    if landing_distances:
        print(f"  최종 평균 착지 거리: {np.mean(landing_distances[-20:]):.1f}px")
        print(f"  최종 성공률 (< 30px): {success_count/episode_count*100:.1f}%")
    print(f"{'='*60}\n")

    # ── 학습 곡선 저장 ────────────────────────────────────────────────────
    pg_plot_path = os.path.join(results_dir, "pg_training_curve.png")
    save_pg_plot(ep_rewards_history, pg_plot_path)
    print(f"REINFORCE 학습 곡선 저장: {pg_plot_path}")

    # ── PPO 결과와 비교 그래프 (PPO 학습이 완료된 경우) ───────────────────
    ppo_history_path = os.path.join(results_dir, "ppo_ep_rewards.npy")
    if os.path.exists(ppo_history_path):
        ppo_rewards = np.load(ppo_history_path).tolist()
        comparison_path = os.path.join(results_dir, "comparison.png")
        save_comparison_plot(ppo_rewards, ep_rewards_history, comparison_path)

    # 비교를 위해 에피소드 보상 저장
    np.save(os.path.join(results_dir, "pg_ep_rewards.npy"), np.array(ep_rewards_history))

    # 모델 가중치 저장
    model_path = os.path.join(results_dir, "pg_drone_drop.pt")
    agent.save(model_path)

    print(f"TensorBoard 로그: {CONFIG['tb_dir']}")
    print(f"  실행: tensorboard --logdir {CONFIG['tb_dir']}")

    return ep_rewards_history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=None, help="총 학습 스텝 수 (기본: CONFIG 값)")
    args = parser.parse_args()
    train_pg(total_timesteps=args.timesteps)
