"""
train.py - PPO CartPole 메인 학습 스크립트

실행:
  python -m cartpole_ppo.train
  또는
  cd "RL Workspace" && python -m cartpole_ppo.train

[학습 사이클]
  while 총 스텝 수 < total_timesteps:
    ① Rollout 수집: 2048 스텝 동안 환경과 상호작용
    ② GAE 계산:    어드밴티지와 returns 계산
    ③ PPO 업데이트: 10 에포크 × 미니배치 학습
    ④ 로깅:        터미널에 학습 상태 출력

[드론 확장 시]
  CONFIG의 env_id, obs_dim, action_dim 만 수정하면
  나머지 PPO 코드는 그대로 재사용 가능.
"""

import os
import time
from collections import deque

import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI 없는 환경에서도 plot 저장 가능
import matplotlib.pyplot as plt
import torch

from .ppo_agent import PPOAgent
from .buffer import RolloutBuffer


# ════════════════════════════════════════════════════════════════════════════
# 하이퍼파라미터 설정
# ════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # ── 환경 설정 ──────────────────────────────────────────────────────────
    # [드론 확장 시] 이 블록만 수정하면 된다.
    "env_id"          : "CartPole-v1",  # 환경 ID
    "obs_dim"         : 4,              # 관측 차원 (CartPole: 카트위치, 속도, 폴각도, 각속도)
    "action_dim"      : 2,              # 행동 개수 (CartPole: 왼쪽=0, 오른쪽=1)

    # ── 수집 설정 ──────────────────────────────────────────────────────────
    "total_timesteps" : 500_000,        # 총 학습 스텝 수
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
    "entropy_coef"    : 0.01,           # 엔트로피 보너스 가중치
    "max_grad_norm"   : 0.5,            # 그래디언트 클리핑 norm

    # ── 신경망 설정 ────────────────────────────────────────────────────────
    "hidden_dim"      : 64,             # 은닉층 크기

    # ── 결과 저장 경로 ─────────────────────────────────────────────────────
    "results_dir"     : "results",
    "model_filename"  : "ppo_cartpole.pt",
    "plot_filename"   : "training_curve.png",

    # ── 로깅 설정 ──────────────────────────────────────────────────────────
    "log_interval"    : 1,              # 매 N rollout마다 로그 출력
    "reward_window"   : 10,             # 이동평균 계산에 사용할 에피소드 수
    "seed"            : 42,
}


def save_training_plot(ep_rewards: list, save_path: str) -> None:
    """
    에피소드 보상 그래프를 저장한다.
    원본 보상과 이동평균을 함께 표시한다.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    episodes = np.arange(1, len(ep_rewards) + 1)
    ax.plot(episodes, ep_rewards, alpha=0.3, color="steelblue", label="Episode Reward")

    # 이동평균 (window=10)
    if len(ep_rewards) >= 10:
        window = 10
        moving_avg = np.convolve(ep_rewards, np.ones(window) / window, mode="valid")
        ax.plot(
            np.arange(window, len(ep_rewards) + 1),
            moving_avg,
            color="steelblue",
            linewidth=2,
            label=f"Moving Average ({window} ep)",
        )

    ax.axhline(y=475, color="green", linestyle="--", alpha=0.7, label="Solved (475)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("PPO CartPole-v1 Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def train(config: dict = CONFIG) -> None:
    """
    PPO CartPole 학습 메인 함수.

    Args:
        config: 하이퍼파라미터 딕셔너리 (기본값: CONFIG)
    """
    # ── 초기화 ──────────────────────────────────────────────────────────────
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    results_dir = config["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  PPO CartPole-v1 학습 시작")
    print(f"  디바이스: {device}")
    print(f"  총 스텝: {config['total_timesteps']:,}")
    print(f"{'='*60}\n")

    # ── 환경 생성 ────────────────────────────────────────────────────────────
    env = gym.make(config["env_id"])
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
    ep_rewards_history = []                        # 모든 에피소드 보상 기록
    recent_rewards = deque(maxlen=config["reward_window"])  # 이동평균용

    timestep = 0
    update_count = 0
    start_time = time.time()

    # ════════════════════════════════════════════════════════════════════════
    # 메인 학습 루프
    # ════════════════════════════════════════════════════════════════════════
    while timestep < total_timesteps:

        # ── ① Rollout 수집 ───────────────────────────────────────────────────
        buffer.reset()

        for _ in range(rollout_length):
            # 현재 정책으로 행동 선택
            action, log_prob, value = agent.select_action(obs)

            # 환경 한 스텝 진행
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 버퍼에 저장
            buffer.store(obs, action, log_prob, reward, value, done)

            obs = next_obs
            ep_reward += reward
            timestep += 1

            # 에피소드 종료 처리
            if done:
                ep_rewards_history.append(ep_reward)
                recent_rewards.append(ep_reward)
                ep_reward = 0.0
                obs, _ = env.reset()

        # ── ② GAE 계산 ──────────────────────────────────────────────────────
        # 롤아웃이 에피소드 중간에 끊겼을 수 있으므로 마지막 V(s)로 bootstrap
        last_value = agent.get_value(obs)
        last_done = done  # 마지막 스텝이 에피소드 종료였는지

        buffer.compute_gae(last_value, last_done)

        # ── ③ PPO 업데이트 ───────────────────────────────────────────────────
        losses = agent.update(buffer, config["n_epochs"], config["batch_size"])
        update_count += 1

        # ── ④ 터미널 로깅 ────────────────────────────────────────────────────
        if update_count % config["log_interval"] == 0 and len(recent_rewards) > 0:
            elapsed = time.time() - start_time
            progress = timestep / total_timesteps * 100
            mean_reward = np.mean(recent_rewards)
            max_reward = np.max(recent_rewards)

            print(f"{'─'*60}")
            print(f"  [Step {timestep:>7,} / {total_timesteps:,}]  진행률: {progress:5.1f}%")
            print(f"  에피소드 수: {len(ep_rewards_history)}")
            print(f"  평균 보상 (최근 {config['reward_window']}개): {mean_reward:7.1f}  |  최대: {max_reward:.0f}")
            print(f"  Actor Loss:  {losses['actor_loss']:8.4f}  |  Critic Loss: {losses['critic_loss']:.4f}")
            print(f"  Entropy:     {losses['entropy']:8.4f}  |  Approx KL:   {losses['approx_kl']:.4f}")
            print(f"  경과 시간: {elapsed:.1f}s")

            # 목표 달성 확인 (CartPole-v1 solved = 평균 475 이상)
            if mean_reward >= 475 and len(recent_rewards) == config["reward_window"]:
                print(f"\n  ✓ 목표 달성! 평균 보상 {mean_reward:.1f} (기준: 475)")

    # ════════════════════════════════════════════════════════════════════════
    # 학습 완료
    # ════════════════════════════════════════════════════════════════════════
    env.close()

    print(f"\n{'='*60}")
    print(f"  학습 완료!")
    print(f"  총 에피소드: {len(ep_rewards_history)}")
    if ep_rewards_history:
        print(f"  최종 평균 보상 (전체): {np.mean(ep_rewards_history):.1f}")
        print(f"  최고 보상: {np.max(ep_rewards_history):.0f}")
    print(f"{'='*60}\n")

    # 모델 저장
    model_path = os.path.join(results_dir, config["model_filename"])
    agent.save(model_path)

    # 학습 곡선 저장
    if ep_rewards_history:
        plot_path = os.path.join(results_dir, config["plot_filename"])
        save_training_plot(ep_rewards_history, plot_path)
        print(f"학습 곡선 저장: {plot_path}")

        # REINFORCE와 비교 그래프를 위해 에피소드 보상 저장
        np.save(os.path.join(results_dir, "ppo_ep_rewards.npy"), np.array(ep_rewards_history))


if __name__ == "__main__":
    train()
