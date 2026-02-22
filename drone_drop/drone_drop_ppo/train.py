"""
train.py - PPO Drone Drop 메인 학습 스크립트

실행:
  cd "RL Workspace/drone_drop"
  python -m drone_drop_ppo.train

[학습 사이클]
  while 총 스텝 수 < total_timesteps:
    ① Rollout 수집: 2048 스텝 동안 환경과 상호작용
    ② GAE 계산:    어드밴티지와 returns 계산
    ③ PPO 업데이트: 10 에포크 × 미니배치 학습
    ④ TensorBoard + 터미널 로깅

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

from .ppo_agent import PPOAgent
from .buffer import RolloutBuffer
from .env import DroneDropEnv


# ════════════════════════════════════════════════════════════════════════════
# 하이퍼파라미터 설정
# ════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # ── 환경 설정 ──────────────────────────────────────────────────────────
    "obs_dim"         : 4,              # 관측 차원 (드론x, 목표x, 상대거리, 투하여부)
    "action_dim"      : 2,              # 행동 개수 (WAIT=0, DROP=1)

    # ── 수집 설정 ──────────────────────────────────────────────────────────
    "total_timesteps" : 300_000,        # 총 학습 스텝 수
    "rollout_length"  : 2048,           # 한 rollout에서 수집할 스텝 수

    # ── PPO 업데이트 설정 ──────────────────────────────────────────────────
    "n_epochs"        : 10,             # rollout당 업데이트 반복 횟수
    "batch_size"      : 64,             # 미니배치 크기

    # ── PPO 핵심 하이퍼파라미터 ────────────────────────────────────────────
    "lr"              : 3e-4,           # Adam 학습률
    "gamma"           : 0.99,           # 할인율 γ
    "lam"             : 0.95,           # GAE lambda λ
    "clip_eps"        : 0.2,            # PPO clip 범위 ε
    "value_coef"      : 0.5,            # Critic 손실 가중치
    "entropy_coef"    : 0.02,           # 엔트로피 보너스 (탐험 장려, cartpole보다 높음)
    "max_grad_norm"   : 0.5,            # 그래디언트 클리핑 norm

    # ── 신경망 설정 ────────────────────────────────────────────────────────
    "hidden_dim"      : 64,             # 은닉층 크기

    # ── 결과 저장 경로 ─────────────────────────────────────────────────────
    "results_dir"     : "results",
    "model_filename"  : "ppo_drone_drop.pt",
    "plot_filename"   : "training_curve.png",
    "tb_dir"          : "results/tensorboard",

    # ── 로깅 설정 ──────────────────────────────────────────────────────────
    "log_interval"    : 1,              # 매 N rollout마다 로그 출력
    "reward_window"   : 20,             # 이동평균 계산에 사용할 에피소드 수
    "seed"            : 42,
}


def save_training_plot(ep_rewards: list, save_path: str) -> None:
    """에피소드 보상 그래프를 저장한다."""
    fig, ax = plt.subplots(figsize=(10, 5))

    episodes = np.arange(1, len(ep_rewards) + 1)
    ax.plot(episodes, ep_rewards, alpha=0.3, color="steelblue", label="Episode Reward")

    if len(ep_rewards) >= 20:
        window = 20
        moving_avg = np.convolve(ep_rewards, np.ones(window) / window, mode="valid")
        ax.plot(
            np.arange(window, len(ep_rewards) + 1),
            moving_avg,
            color="steelblue",
            linewidth=2,
            label=f"Moving Average ({window} ep)",
        )

    ax.axhline(y=12.0, color="green", linestyle="--", alpha=0.7, label="Target (12.0)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("PPO Drone Drop Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def train(config: dict = CONFIG) -> None:
    """PPO Drone Drop 학습 메인 함수."""
    # ── 초기화 ──────────────────────────────────────────────────────────────
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    results_dir = config["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  PPO Drone Drop 학습 시작")
    print(f"  디바이스: {device}")
    print(f"  총 스텝: {config['total_timesteps']:,}")
    print(f"{'='*60}\n")

    # ── TensorBoard ─────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=os.path.join(config["tb_dir"], "ppo"))

    # ── 환경 생성 ────────────────────────────────────────────────────────────
    env = DroneDropEnv()
    env.reset(seed=config["seed"])

    # ── PPO 에이전트 + 롤아웃 버퍼 ──────────────────────────────────────────
    agent = PPOAgent(
        obs_dim=config["obs_dim"],
        action_dim=config["action_dim"],
        hidden_dim=config["hidden_dim"],
        lr=config["lr"],
        clip_eps=config["clip_eps"],
        value_coef=config["value_coef"],
        entropy_coef=config["entropy_coef"],
        max_grad_norm=config["max_grad_norm"],
        device=device,
    )

    buffer = RolloutBuffer(
        rollout_length=config["rollout_length"],
        obs_dim=config["obs_dim"],
        gamma=config["gamma"],
        lam=config["lam"],
        device=device,
    )

    # ── 학습 추적 변수 ────────────────────────────────────────────────────────
    total_timesteps = config["total_timesteps"]
    rollout_length = config["rollout_length"]

    obs, _ = env.reset()
    ep_reward = 0.0
    ep_rewards_history = []
    recent_rewards = deque(maxlen=config["reward_window"])

    # 환경 특화 지표 추적
    landing_distances = []     # 착지 거리 기록
    drop_timings = []          # 투하 시점 (drone_x) 기록
    success_count = 0          # 정밀 착지 (< 30px) 횟수
    episode_count = 0          # 총 에피소드 수

    timestep = 0
    update_count = 0
    start_time = time.time()

    # ════════════════════════════════════════════════════════════════════════
    # 메인 학습 루프
    # ════════════════════════════════════════════════════════════════════════
    while timestep < total_timesteps:

        # ── ① Rollout 수집 ───────────────────────────────────────────────────
        buffer.reset()
        rollout_distances = []
        rollout_drop_timings = []

        for _ in range(rollout_length):
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer.store(obs, action, log_prob, reward, value, done)

            obs = next_obs
            ep_reward += reward
            timestep += 1

            if done:
                ep_rewards_history.append(ep_reward)
                recent_rewards.append(ep_reward)
                episode_count += 1

                # 환경 특화 지표 수집
                if info.get("distance") is not None:
                    dist = info["distance"]
                    landing_distances.append(dist)
                    rollout_distances.append(dist)
                    if dist < 30.0:
                        success_count += 1

                    # TensorBoard: 에피소드별 지표
                    writer.add_scalar("Reward/episode_reward", ep_reward, episode_count)
                    writer.add_scalar("Environment/landing_distance", dist, episode_count)

                if info.get("drop_x") is not None:
                    drop_timings.append(info["drop_x"])
                    rollout_drop_timings.append(info["drop_x"])
                    writer.add_scalar("Environment/drop_timing", info["drop_x"], episode_count)

                ep_reward = 0.0
                obs, _ = env.reset()

        # ── ② GAE 계산 ──────────────────────────────────────────────────────
        last_value = agent.get_value(obs)
        last_done = done
        buffer.compute_gae(last_value, last_done)

        # ── ③ PPO 업데이트 ───────────────────────────────────────────────────
        losses = agent.update(buffer, config["n_epochs"], config["batch_size"])
        update_count += 1

        # ── ④ TensorBoard 로깅 ──────────────────────────────────────────────
        writer.add_scalar("Loss/actor_loss", losses["actor_loss"], update_count)
        writer.add_scalar("Loss/critic_loss", losses["critic_loss"], update_count)
        writer.add_scalar("Loss/entropy", losses["entropy"], update_count)
        writer.add_scalar("Loss/approx_kl", losses["approx_kl"], update_count)

        if len(recent_rewards) > 0:
            mean_reward = np.mean(recent_rewards)
            writer.add_scalar("Reward/mean_reward", mean_reward, update_count)

            if episode_count > 0:
                writer.add_scalar(
                    "Environment/success_rate",
                    success_count / episode_count * 100,
                    update_count,
                )

            if rollout_distances:
                writer.add_scalar(
                    "Environment/mean_landing_distance",
                    np.mean(rollout_distances),
                    update_count,
                )

        # ── ⑤ 터미널 로깅 ────────────────────────────────────────────────────
        if update_count % config["log_interval"] == 0 and len(recent_rewards) > 0:
            elapsed = time.time() - start_time
            progress = timestep / total_timesteps * 100
            mean_reward = np.mean(recent_rewards)
            max_reward = np.max(recent_rewards)

            print(f"{'─'*60}")
            print(f"  [Step {timestep:>7,} / {total_timesteps:,}]  진행률: {progress:5.1f}%")
            print(f"  에피소드 수: {episode_count}")
            print(f"  평균 보상 (최근 {config['reward_window']}개): {mean_reward:7.2f}  |  최대: {max_reward:.2f}")
            print(f"  Actor Loss:  {losses['actor_loss']:8.4f}  |  Critic Loss: {losses['critic_loss']:.4f}")
            print(f"  Entropy:     {losses['entropy']:8.4f}  |  Approx KL:   {losses['approx_kl']:.4f}")

            if landing_distances:
                recent_dist = landing_distances[-min(20, len(landing_distances)):]
                print(f"  평균 착지 거리: {np.mean(recent_dist):7.1f}px  |  성공률: {success_count/episode_count*100:.1f}%")

            print(f"  경과 시간: {elapsed:.1f}s")

            if mean_reward >= 12.0 and len(recent_rewards) == config["reward_window"]:
                print(f"\n  ✓ 목표 달성! 평균 보상 {mean_reward:.2f} (기준: 12.0)")

    # ════════════════════════════════════════════════════════════════════════
    # 학습 완료
    # ════════════════════════════════════════════════════════════════════════
    env.close()
    writer.close()

    print(f"\n{'='*60}")
    print(f"  학습 완료!")
    print(f"  총 에피소드: {episode_count}")
    if ep_rewards_history:
        print(f"  최종 평균 보상 (전체): {np.mean(ep_rewards_history):.2f}")
        print(f"  최고 보상: {np.max(ep_rewards_history):.2f}")
    if landing_distances:
        print(f"  최종 평균 착지 거리: {np.mean(landing_distances[-20:]):.1f}px")
        print(f"  최종 성공률 (< 30px): {success_count/episode_count*100:.1f}%")
    print(f"{'='*60}\n")

    # 모델 저장
    model_path = os.path.join(results_dir, config["model_filename"])
    agent.save(model_path)

    # 학습 곡선 저장
    if ep_rewards_history:
        plot_path = os.path.join(results_dir, config["plot_filename"])
        save_training_plot(ep_rewards_history, plot_path)
        # REINFORCE와 비교 그래프를 위해 에피소드 보상 저장
        np.save(os.path.join(results_dir, "ppo_ep_rewards.npy"), np.array(ep_rewards_history))

        print(f"학습 곡선 저장: {plot_path}")
        print(f"TensorBoard 로그: {config['tb_dir']}")
        print(f"  실행: tensorboard --logdir {config['tb_dir']}")


if __name__ == "__main__":
    train()
