"""
compare.py — Shaped vs Sparse 보상 비교 분석

저장된 보상/거리 기록(.npy)을 읽어 비교 차트를 생성한다.

실행:
  cd "RL Workspace"
  python -m drone_drop_rl.compare
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .configs.base import BASE_CONFIG

ALGO_COLORS = {
    "sac_shaped": ("steelblue", "SAC + Shaped Reward"),
    "sac_sparse": ("coral", "SAC + Sparse Reward"),
}


def load_data():
    """저장된 보상/거리 기록을 로드한다."""
    plots_dir = BASE_CONFIG["plots_dir"]
    data = {}
    for algo_name in ALGO_COLORS:
        reward_path = os.path.join(plots_dir, f"{algo_name}_rewards.npy")
        dist_path = os.path.join(plots_dir, f"{algo_name}_distances.npy")

        if os.path.exists(reward_path):
            entry = {"rewards": np.load(reward_path)}
            if os.path.exists(dist_path):
                entry["distances"] = np.load(dist_path)
            data[algo_name] = entry
            print(f"  로드: {algo_name} ({len(entry['rewards'])} 에피소드)")
        else:
            print(f"  없음: {algo_name} — 먼저 학습하세요")
    return data


def plot_comparison(data: dict, save_dir: str):
    """비교 차트를 생성한다."""
    os.makedirs(save_dir, exist_ok=True)

    if len(data) < 2:
        print("\n  비교를 위해 shaped와 sparse 둘 다 학습 결과가 필요합니다.")
        return

    # ── 1. 학습 곡선 비교 ─────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1-1. 보상 학습 곡선
    ax = axes[0, 0]
    for algo, entry in data.items():
        rewards = entry["rewards"]
        color, label = ALGO_COLORS[algo]
        ax.plot(rewards, alpha=0.1, color=color)
        if len(rewards) >= 50:
            window = 50
            ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(np.arange(window - 1, len(rewards)), ma,
                    color=color, linewidth=2, label=label)
    ax.axhline(y=15, color="green", linestyle="--", alpha=0.5, label="Target (15)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("학습 곡선 비교 — Shaped vs Sparse")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 1-2. 착지 거리 학습 곡선
    ax = axes[0, 1]
    for algo, entry in data.items():
        if "distances" in entry:
            dists = entry["distances"]
            color, label = ALGO_COLORS[algo]
            ax.plot(dists, alpha=0.1, color=color)
            if len(dists) >= 50:
                window = 50
                ma = np.convolve(dists, np.ones(window) / window, mode="valid")
                ax.plot(np.arange(window - 1, len(dists)), ma,
                        color=color, linewidth=2, label=label)
    ax.axhline(y=5, color="green", linestyle="--", alpha=0.5, label="5m 기준")
    ax.set_xlabel("Drop Episode")
    ax.set_ylabel("Landing Distance (m)")
    ax.set_title("착지 거리 변화 — Shaped vs Sparse")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 1-3. 보상 분포 (마지막 100 에피소드)
    ax = axes[1, 0]
    labels = []
    box_data = []
    colors = []
    for algo in ALGO_COLORS:
        if algo in data:
            rewards = data[algo]["rewards"]
            last_n = min(100, len(rewards))
            box_data.append(rewards[-last_n:])
            color, label = ALGO_COLORS[algo]
            labels.append(label)
            colors.append(color)

    bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Total Reward")
    ax.set_title("최종 성능 분포 (Last 100 Episodes)")
    ax.grid(True, alpha=0.3, axis="y")

    # 1-4. 착지 거리 분포
    ax = axes[1, 1]
    labels = []
    box_data = []
    colors = []
    for algo in ALGO_COLORS:
        if algo in data and "distances" in data[algo]:
            dists = data[algo]["distances"]
            last_n = min(100, len(dists))
            box_data.append(dists[-last_n:])
            color, label = ALGO_COLORS[algo]
            labels.append(label)
            colors.append(color)

    if box_data:
        bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax.axhline(y=5, color="green", linestyle="--", alpha=0.5, label="5m 기준")
    ax.set_ylabel("Landing Distance (m)")
    ax.set_title("착지 거리 분포 (Last 100 Drops)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Shaped vs Sparse Reward — 비교 분석", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "comparison_shaped_vs_sparse.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  차트 저장: {path}")

    # ── 2. 요약 테이블 ────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  {'보상 타입':<25s} {'에피소드':>8s} {'평균 보상':>10s} "
          f"{'평균 거리':>10s} {'성공률(<5m)':>12s}")
    print(f"  {'─'*65}")
    for algo in ALGO_COLORS:
        if algo in data:
            entry = data[algo]
            rewards = entry["rewards"]
            _, label = ALGO_COLORS[algo]
            last_rewards = rewards[-100:] if len(rewards) >= 100 else rewards

            if "distances" in entry:
                dists = entry["distances"]
                last_dists = dists[-100:] if len(dists) >= 100 else dists
                mean_dist = f"{np.mean(last_dists):>10.2f}"
                success_rate = f"{np.mean(last_dists < 5.0):>11.1%}"
            else:
                mean_dist = f"{'N/A':>10s}"
                success_rate = f"{'N/A':>11s}"

            print(f"  {label:<25s} {len(rewards):>8d} {np.mean(last_rewards):>10.2f} "
                  f"{mean_dist} {success_rate}")
    print(f"{'='*75}")


def main():
    print(f"\n{'='*60}")
    print(f"  Drone Drop — Shaped vs Sparse 보상 비교 분석")
    print(f"{'='*60}\n")

    data = load_data()
    plot_comparison(data, BASE_CONFIG["plots_dir"])


if __name__ == "__main__":
    main()
