"""
compare.py — 알고리즘 비교 차트 생성

저장된 보상 기록(.npy)을 읽어 비교 분석 차트를 생성한다.

실행:
  cd "RL Workspace"
  python -m lunar_lander.compare
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .configs.base import BASE_CONFIG

ALGO_COLORS = {
    "ppo_discrete": ("steelblue", "PPO (Discrete)"),
    "reinforce": ("coral", "REINFORCE"),
    "dqn": ("mediumpurple", "DQN"),
    "ppo_continuous": ("teal", "PPO (Continuous)"),
    "sac": ("darkorange", "SAC"),
}


def load_rewards():
    """저장된 보상 기록을 로드한다."""
    plots_dir = BASE_CONFIG["plots_dir"]
    data = {}
    for algo_name in ALGO_COLORS:
        path = os.path.join(plots_dir, f"{algo_name}_rewards.npy")
        if os.path.exists(path):
            data[algo_name] = np.load(path)
            print(f"  로드: {algo_name} ({len(data[algo_name])} 에피소드)")
        else:
            print(f"  없음: {algo_name} — 먼저 학습하세요")
    return data


def plot_comparison(data: dict, save_dir: str):
    """비교 차트를 생성한다."""
    os.makedirs(save_dir, exist_ok=True)

    # ── 1. 이산 환경 학습 곡선 비교 ─────────────────────────────────────────
    discrete_algos = ["ppo_discrete", "reinforce", "dqn"]
    discrete_data = {k: v for k, v in data.items() if k in discrete_algos}

    if len(discrete_data) > 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        for algo, rewards in discrete_data.items():
            color, label = ALGO_COLORS[algo]
            ax.plot(rewards, alpha=0.15, color=color)
            if len(rewards) >= 20:
                window = 20
                ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
                ax.plot(np.arange(window - 1, len(rewards)), ma, color=color, linewidth=2, label=label)
        ax.axhline(y=200, color="green", linestyle="--", alpha=0.5, label="Solved (200)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title("LunarLander-v3 (Discrete) — Algorithm Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(save_dir, "comparison_discrete.png")
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"\n  차트 저장: {path}")

    # ── 2. 연속 환경 학습 곡선 비교 ─────────────────────────────────────────
    continuous_algos = ["ppo_continuous", "sac"]
    continuous_data = {k: v for k, v in data.items() if k in continuous_algos}

    if len(continuous_data) > 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        for algo, rewards in continuous_data.items():
            color, label = ALGO_COLORS[algo]
            ax.plot(rewards, alpha=0.15, color=color)
            if len(rewards) >= 20:
                window = 20
                ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
                ax.plot(np.arange(window - 1, len(rewards)), ma, color=color, linewidth=2, label=label)
        ax.axhline(y=200, color="green", linestyle="--", alpha=0.5, label="Solved (200)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title("LunarLanderContinuous-v3 — Algorithm Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(save_dir, "comparison_continuous.png")
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  차트 저장: {path}")

    # ── 3. 전체 5개 알고리즘 박스플롯 ───────────────────────────────────────
    if len(data) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = []
        box_data = []
        colors = []
        for algo in ALGO_COLORS:
            if algo in data:
                rewards = data[algo]
                # 마지막 50 에피소드의 분포
                last_n = min(50, len(rewards))
                box_data.append(rewards[-last_n:])
                color, label = ALGO_COLORS[algo]
                labels.append(label)
                colors.append(color)

        bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.axhline(y=200, color="green", linestyle="--", alpha=0.5, label="Solved (200)")
        ax.set_ylabel("Total Reward")
        ax.set_title("Final Performance Distribution (Last 50 Episodes)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        path = os.path.join(save_dir, "comparison_boxplot.png")
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  차트 저장: {path}")

    # ── 4. 요약 테이블 출력 ─────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  {'알고리즘':<22s} {'에피소드':>8s} {'최종 평균':>10s} {'최고':>8s} {'최저':>8s}")
    print(f"  {'─'*56}")
    for algo in ALGO_COLORS:
        if algo in data:
            rewards = data[algo]
            last_20 = rewards[-20:] if len(rewards) >= 20 else rewards
            _, label = ALGO_COLORS[algo]
            print(f"  {label:<22s} {len(rewards):>8d} {np.mean(last_20):>10.1f} "
                  f"{np.max(rewards):>8.1f} {np.min(rewards):>8.1f}")
    print(f"{'='*65}")


def main():
    print(f"\n{'='*60}")
    print(f"  LunarLander — 알고리즘 비교 분석")
    print(f"{'='*60}\n")

    data = load_rewards()
    if len(data) < 2:
        print("\n  비교를 위해 최소 2개 알고리즘의 학습 결과가 필요합니다.")
        return

    plot_comparison(data, BASE_CONFIG["plots_dir"])


if __name__ == "__main__":
    main()
