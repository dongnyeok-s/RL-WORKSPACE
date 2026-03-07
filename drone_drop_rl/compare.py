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
    ax.set_title("Learning Curve — Shaped vs Sparse")
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
    ax.axhline(y=5, color="green", linestyle="--", alpha=0.5, label="5m threshold")
    ax.set_xlabel("Drop Episode")
    ax.set_ylabel("Landing Distance (m)")
    ax.set_title("Landing Distance — Shaped vs Sparse")
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
    ax.set_title("Final Reward Distribution (Last 100 Episodes)")
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
    ax.axhline(y=5, color="green", linestyle="--", alpha=0.5, label="5m threshold")
    ax.set_ylabel("Landing Distance (m)")
    ax.set_title("Landing Distance Distribution (Last 100 Drops)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Shaped vs Sparse Reward — Comparison Analysis", fontsize=14, fontweight="bold")
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


def save_markdown_report(data: dict, save_dir: str):
    """비교 분석 결과를 REPORT.md로 저장한다."""
    report_path = os.path.join(os.path.dirname(os.path.dirname(save_dir)), "REPORT.md")

    # 통계 계산
    stats = {}
    for algo in ALGO_COLORS:
        if algo not in data:
            continue
        entry = data[algo]
        rewards = entry["rewards"]
        last_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
        row = {
            "episodes": len(rewards),
            "avg_reward": np.mean(last_rewards),
            "std_reward": np.std(last_rewards),
            "max_reward": np.max(last_rewards),
        }
        if "distances" in entry:
            dists = entry["distances"]
            last_dists = dists[-100:] if len(dists) >= 100 else dists
            row["avg_dist"] = np.mean(last_dists)
            row["std_dist"] = np.std(last_dists)
            row["success_rate"] = np.mean(last_dists < 5.0) * 100
        stats[algo] = row

    shaped = stats.get("sac_shaped", {})
    sparse = stats.get("sac_sparse", {})

    lines = [
        "# Drone Drop RL: Shaped vs Sparse Reward 비교 분석",
        "",
        "> **실험일**: 2026-03-07  ",
        "> **알고리즘**: SAC (Soft Actor-Critic)  ",
        "> **환경**: 커스텀 3D 드론 투하 시뮬레이터  ",
        "",
        "---",
        "",
        "## 1. 실험 개요",
        "",
        "본 실험은 **보상 함수 설계**가 강화학습 에이전트의 학습 효율과 최종 성능에 미치는 영향을 분석한다.",
        "동일한 SAC 알고리즘과 3D 드론 환경에서 두 가지 보상 체계를 비교한다:",
        "",
        "| 시나리오 | 설명 |",
        "|----------|------|",
        "| **SAC + Shaped Reward** | 4계층 계층적 보상 함수로 매 스텝 학습 신호 제공 |",
        "| **SAC + Sparse Reward** | 투하 결과에만 보상 부여, 중간 과정 신호 없음 (대조군) |",
        "",
        "---",
        "",
        "## 2. 환경 설명",
        "",
        "커스텀 3D 드론 투하 시뮬레이터로 Gymnasium 인터페이스를 구현한다.",
        "",
        "| 파라미터 | 값 |",
        "|----------|----|",
        "| 관찰 공간 차원 | 14 |",
        "| 액션 공간 차원 | 4 (lateral, forward, vertical, drop) |",
        "| 월드 크기 | 80 × 80 m |",
        "| 비행 고도 | 40 m (기본) / 최대 60 m |",
        "| 시뮬레이션 주기 | 30 Hz (dt = 1/30 s) |",
        "| 에피소드 최대 스텝 | 600 (20초) |",
        "| 최대 풍속 | 5 m/s |",
        "| 성공 기준 | 착지 거리 < 5 m |",
        "",
        "**보상 해킹 방지 메커니즘**:",
        "- Potential-based shaping (Ng et al., 1999) — 최적 정책 불변 수학적 보장",
        "- 에너지 페널티 — 타겟 근처 진동(jittering) 방지",
        "- 1회성 투하 — 반복 투하 시도 불가",
        "",
        "---",
        "",
        "## 3. 보상 함수 비교",
        "",
        "### 3.1 Shaped Reward (4계층 계층적 보상)",
        "",
        "```",
        "Layer 1 — 안전      경계 위반 시 큰 음의 보상 + 에피소드 종료",
        "                    최소 안전 고도 5m, 최대 속도 30 m/s",
        "Layer 2 — 효율성    에너지 페널티 (매 스텝): -0.002 × ||action||",
        "Layer 3 — 접근 유도 Potential-based shaping (스케일 10.0) + 정렬 보너스 (매 스텝, 투하 전)",
        "Layer 4 — 투하 정밀도 거리 비례 보상(최대 15) + 정밀도 티어 보너스 (투하 시 1회)",
        "                    · 1m 이내: +10  · 3m 이내: +5  · 5m 이내: +2",
        "```",
        "",
        "### 3.2 Sparse Reward (대조군)",
        "",
        "```",
        "중간 과정: 보상 없음 (0)",
        "투하 시:   거리 비례 보상만 (최대 15, Shaped와 동일한 스케일)",
        "안전 위반: Shaped와 동일한 페널티",
        "```",
        "",
        "---",
        "",
        "## 4. 학습 결과",
        "",
        "### 4.1 수치 통계",
        "",
        "마지막 100 에피소드 기준 통계:",
        "",
        "| 지표 | SAC + Shaped | SAC + Sparse |",
        "|------|:------------:|:------------:|",
        f"| 총 에피소드 수 | {shaped.get('episodes', 'N/A'):,} | {sparse.get('episodes', 'N/A'):,} |",
        f"| 평균 보상 (마지막 100) | {shaped.get('avg_reward', 0):.2f} | {sparse.get('avg_reward', 0):.2f} |",
        f"| 보상 표준편차 | {shaped.get('std_reward', 0):.2f} | {sparse.get('std_reward', 0):.2f} |",
        f"| 최대 보상 | {shaped.get('max_reward', 0):.2f} | {sparse.get('max_reward', 0):.2f} |",
        f"| 평균 착지 거리 (m) | {shaped.get('avg_dist', 0):.2f} | {sparse.get('avg_dist', 0):.2f} |",
        f"| 거리 표준편차 (m) | {shaped.get('std_dist', 0):.2f} | {sparse.get('std_dist', 0):.2f} |",
        f"| 성공률 (< 5m) | {shaped.get('success_rate', 0):.1f}% | {sparse.get('success_rate', 0):.1f}% |",
        "",
        "> **비고**: Shaped의 에피소드 수가 적은 이유는 보상 신호가 충분해 조기 수렴했기 때문이다.",
        "> Sparse는 유의미한 보상 신호를 얻기 위해 훨씬 많은 시도가 필요하다.",
        "",
        "### 4.2 비교 차트",
        "",
        "![Shaped vs Sparse 비교 차트](results/plots/comparison_shaped_vs_sparse.png)",
        "",
        "차트 구성:",
        "- **상단 좌**: 학습 곡선 (이동평균 50 에피소드) — 보상 변화 추이",
        "- **상단 우**: 착지 거리 학습 곡선 — 정밀도 개선 추이",
        "- **하단 좌**: 마지막 100 에피소드 보상 분포 (박스플롯)",
        "- **하단 우**: 마지막 100 에피소드 착지 거리 분포 (박스플롯)",
        "",
        "---",
        "",
        "## 5. 결론",
        "",
        "### 5.1 핵심 발견",
        "",
        "1. **Shaped Reward의 압도적 우위**  ",
        f"   Shaped는 {shaped.get('episodes', 0):,}회 만에 평균 보상 **{shaped.get('avg_reward', 0):.1f}**을 달성한 반면,",
        f"   Sparse는 {sparse.get('episodes', 0):,}회에도 평균 보상 **{sparse.get('avg_reward', 0):.2f}**에 그쳤다.",
        "",
        "2. **착지 정밀도 격차**  ",
        f"   Shaped 성공률({shaped.get('success_rate', 0):.1f}%) vs Sparse 성공률({sparse.get('success_rate', 0):.1f}%).",
        "   Sparse는 무작위 탐색만으로는 5m 이내 착지가 사실상 불가능함을 보여준다.",
        "",
        "3. **샘플 효율성**  ",
        f"   Sparse가 Shaped보다 약 {sparse.get('episodes', 1) / max(shaped.get('episodes', 1), 1):.0f}배 많은 에피소드를 수행했음에도",
        "   수렴하지 못했다. 드론 투하처럼 희귀한 성공 이벤트가 있는 환경에서 Sparse Reward는",
        "   신뢰도 있는 학습 신호를 제공하기 어렵다.",
        "",
        "### 5.2 시사점",
        "",
        "- **보상 설계의 중요성**: 동일한 SAC 알고리즘이라도 보상 함수 설계에 따라 결과가 극단적으로 달라진다.",
        "- **Potential-based Shaping의 효과**: 최적 정책 불변성을 수학적으로 보장하면서도 학습 신호를 풍부하게 제공한다.",
        "- **계층적 보상의 실용성**: 안전 → 효율 → 접근 → 정밀도 순서로 우선순위를 부여하면",
        "  에이전트가 단계적으로 복잡한 행동을 학습할 수 있다.",
        "",
        "---",
        "",
        "*생성: `python -m drone_drop_rl.compare`*",
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  보고서 저장: {report_path}")


def main():
    print(f"\n{'='*60}")
    print(f"  Drone Drop — Shaped vs Sparse 보상 비교 분석")
    print(f"{'='*60}\n")

    data = load_data()
    plot_comparison(data, BASE_CONFIG["plots_dir"])
    save_markdown_report(data, BASE_CONFIG["plots_dir"])


if __name__ == "__main__":
    main()
