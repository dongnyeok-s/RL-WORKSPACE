"""
비교 분석 그래프 생성 스크립트
PPO vs REINFORCE 학습 결과를 타임스텝 기준으로 공정하게 비교한다.

핵심 이슈:
  - REINFORCE: 에피소드 ~300K개 (대부분 1스텝, 즉시 DROP)
  - PPO: 에피소드 ~14K개 (rollout 기반, 에피소드가 더 길어짐)
  → 에피소드 기준 비교는 불공정하므로 "타임스텝" 기준으로 비교한다.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# ── 한글 폰트 설정 ────────────────────────────────────────────────────────
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

RESULTS_DIR = "results"
TOTAL_TIMESTEPS = 300_000

# ── 데이터 로드 ──────────────────────────────────────────────────────────
ppo_rewards = np.load(os.path.join(RESULTS_DIR, "ppo_ep_rewards.npy"))
pg_rewards = np.load(os.path.join(RESULTS_DIR, "pg_ep_rewards.npy"))

print(f"PPO      : {len(ppo_rewards):,} episodes, mean={np.mean(ppo_rewards):.2f}")
print(f"REINFORCE: {len(pg_rewards):,} episodes, mean={np.mean(pg_rewards):.2f}")

# ── 타임스텝 복원 ────────────────────────────────────────────────────────
# 에피소드 보상으로부터 각 에피소드의 스텝 수를 추정한다.
# DroneDropEnv: 시간 페널티 -0.01/step, 투하 보상 또는 미투하 패널티
# 에피소드 보상 = (투하/미투하 보상) + (스텝 수 * -0.01)
# → 스텝 수 = 보상이 양수면 빠른 투하(짧은 에피소드), 음수면 늦은 투하/미투하


def estimate_timesteps(rewards: np.ndarray, total: int) -> np.ndarray:
    """에피소드 보상으로부터 누적 타임스텝을 추정한다.

    정확한 스텝 수는 저장되어 있지 않으므로,
    총 타임스텝(300K)을 에피소드 수에 맞게 균등 배분한다.
    """
    n = len(rewards)
    return np.linspace(0, total, n, endpoint=True)


ppo_ts = estimate_timesteps(ppo_rewards, TOTAL_TIMESTEPS)
pg_ts = estimate_timesteps(pg_rewards, TOTAL_TIMESTEPS)


def moving_average_ts(timesteps, rewards, window_ts=10000):
    """타임스텝 기준 이동평균을 계산한다.

    고정 타임스텝 윈도우로 보상을 평균하여
    에피소드 수 차이와 관계없이 공정하게 비교할 수 있다.
    """
    result_ts = []
    result_avg = []
    for i in range(len(timesteps)):
        t = timesteps[i]
        # 현재 시점에서 window_ts 이내의 에피소드들을 평균
        mask = (timesteps >= t - window_ts) & (timesteps <= t)
        if mask.sum() >= 5:  # 최소 5개 에피소드
            result_ts.append(t)
            result_avg.append(np.mean(rewards[mask]))
    return np.array(result_ts), np.array(result_avg)


# ════════════════════════════════════════════════════════════════════════════
# 색상 팔레트
# ════════════════════════════════════════════════════════════════════════════
PPO_COLOR = "#1565C0"       # Deep Blue
PPO_LIGHT = "#BBDEFB"
PG_COLOR = "#E64A19"        # Deep Orange
PG_LIGHT = "#FFCCBC"
TARGET_COLOR = "#2E7D32"    # Deep Green
ACCENT = "#7B1FA2"          # Purple accent


# ════════════════════════════════════════════════════════════════════════════
# Figure 1: 종합 비교 대시보드 (2x2)
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor("#FAFAFA")

gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30,
                       left=0.07, right=0.96, top=0.89, bottom=0.07)

# ── (1) 학습 곡선 비교 (타임스텝 기준) ────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("#FFFFFF")

# 원시 데이터 (투명하게)
ax1.scatter(ppo_ts, ppo_rewards, alpha=0.03, color=PPO_COLOR, s=2, rasterized=True)
ax1.scatter(pg_ts, pg_rewards, alpha=0.03, color=PG_COLOR, s=1, rasterized=True)

# 타임스텝 기준 이동평균 (10K steps window)
ppo_ma_ts, ppo_ma = moving_average_ts(ppo_ts, ppo_rewards, window_ts=15000)
pg_ma_ts, pg_ma = moving_average_ts(pg_ts, pg_rewards, window_ts=15000)

ax1.plot(ppo_ma_ts, ppo_ma, color=PPO_COLOR, linewidth=2.5, label="PPO", zorder=5)
ax1.plot(pg_ma_ts, pg_ma, color=PG_COLOR, linewidth=2.5, label="REINFORCE", zorder=5)

ax1.axhline(y=12.0, color=TARGET_COLOR, linestyle="--", alpha=0.5, linewidth=1.2, label="Target (12.0)")
ax1.set_xlabel("Timestep", fontsize=11)
ax1.set_ylabel("Episode Reward", fontsize=11)
ax1.set_title("Learning Curve (Timestep-aligned)", fontsize=13, fontweight="bold", pad=10)
ax1.legend(loc="lower right", fontsize=10, framealpha=0.9)
ax1.grid(True, alpha=0.15)
ax1.set_ylim(-6, 16)
ax1.set_xlim(0, TOTAL_TIMESTEPS)
# x축 포맷 (K 단위)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

# ── (2) 보상 분포 비교 (Violin + Box) ──────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor("#FFFFFF")

# 후반 50% 데이터 (수렴 후 성능)
ppo_late = ppo_rewards[len(ppo_rewards)//2:]
pg_late = pg_rewards[len(pg_rewards)//2:]

vp = ax2.violinplot([ppo_late, pg_late], positions=[1, 2],
                     showmedians=False, showextrema=False)
for i, body in enumerate(vp["bodies"]):
    body.set_facecolor([PPO_LIGHT, PG_LIGHT][i])
    body.set_edgecolor([PPO_COLOR, PG_COLOR][i])
    body.set_alpha(0.6)
    body.set_linewidth(1.5)

bp = ax2.boxplot([ppo_late, pg_late], positions=[1, 2], widths=0.18,
                 patch_artist=True, showfliers=False,
                 medianprops=dict(color="white", linewidth=2.5),
                 whiskerprops=dict(linewidth=1.5),
                 capprops=dict(linewidth=1.5))
for i, patch in enumerate(bp["boxes"]):
    patch.set_facecolor([PPO_COLOR, PG_COLOR][i])
    patch.set_alpha(0.85)

ax2.axhline(y=12.0, color=TARGET_COLOR, linestyle="--", alpha=0.5, linewidth=1.2)

ax2.set_xticks([1, 2])
ax2.set_xticklabels(["PPO", "REINFORCE"], fontsize=12, fontweight="bold")
ax2.set_ylabel("Reward", fontsize=11)
ax2.set_title("Reward Distribution (Last 50% of Training)", fontsize=13, fontweight="bold", pad=10)
ax2.grid(True, alpha=0.15, axis="y")

# 중앙값/평균 텍스트
for i, (data, color) in enumerate([(ppo_late, PPO_COLOR), (pg_late, PG_COLOR)]):
    med = np.median(data)
    mean = np.mean(data)
    ax2.text(i+1, 16.5, f"mean={mean:.1f}\nmed={med:.1f}",
             ha="center", va="top", fontsize=9, color=color,
             fontweight="bold", linespacing=1.4,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, alpha=0.8))

ax2.set_ylim(-6, 17)

# ── (3) 학습 안정성 비교 (Rolling Std) ────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor("#FFFFFF")

# 타임스텝 기준 Rolling Std
def rolling_std_ts(timesteps, rewards, window_ts=15000):
    result_ts = []
    result_std = []
    for i in range(len(timesteps)):
        t = timesteps[i]
        mask = (timesteps >= t - window_ts) & (timesteps <= t)
        if mask.sum() >= 5:
            result_ts.append(t)
            result_std.append(np.std(rewards[mask]))
    return np.array(result_ts), np.array(result_std)

ppo_std_ts, ppo_std = rolling_std_ts(ppo_ts, ppo_rewards)
pg_std_ts, pg_std = rolling_std_ts(pg_ts, pg_rewards)

ax3.fill_between(ppo_std_ts, ppo_std, alpha=0.15, color=PPO_COLOR)
ax3.fill_between(pg_std_ts, pg_std, alpha=0.15, color=PG_COLOR)
ax3.plot(ppo_std_ts, ppo_std, color=PPO_COLOR, linewidth=2, label="PPO")
ax3.plot(pg_std_ts, pg_std, color=PG_COLOR, linewidth=2, label="REINFORCE")

ax3.set_xlabel("Timestep", fontsize=11)
ax3.set_ylabel("Reward Std Dev", fontsize=11)
ax3.set_title("Learning Stability (Lower = More Stable)", fontsize=13, fontweight="bold", pad=10)
ax3.legend(loc="upper right", fontsize=10, framealpha=0.9)
ax3.grid(True, alpha=0.15)
ax3.set_xlim(0, TOTAL_TIMESTEPS)
ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

# ── (4) 핵심 지표 요약 테이블 ──────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor("#FFFFFF")
ax4.axis("off")

# 지표 계산
ppo_success_rate = np.mean(ppo_rewards > 12.0) * 100
pg_success_rate = np.mean(pg_rewards > 12.0) * 100
ppo_perfect_rate = np.mean(ppo_rewards > 14.0) * 100
pg_perfect_rate = np.mean(pg_rewards > 14.0) * 100

metrics = [
    ("Total Timesteps", "300,000", "300,000", None),
    ("Total Episodes", f"{len(ppo_rewards):,}", f"{len(pg_rewards):,}", None),
    ("Avg Steps/Episode", f"{TOTAL_TIMESTEPS/len(ppo_rewards):.1f}", f"{TOTAL_TIMESTEPS/len(pg_rewards):.1f}", None),
    ("Mean Reward", f"{np.mean(ppo_rewards):.2f}", f"{np.mean(pg_rewards):.2f}", "ppo"),
    ("Max Reward", f"{np.max(ppo_rewards):.2f}", f"{np.max(pg_rewards):.2f}", "ppo"),
    ("Median (last 50%)", f"{np.median(ppo_late):.2f}", f"{np.median(pg_late):.2f}", "ppo"),
    ("Std Dev (last 50%)", f"{np.std(ppo_late):.2f}", f"{np.std(pg_late):.2f}", "pg"),
    ("Success Rate (>12)", f"{ppo_success_rate:.1f}%", f"{pg_success_rate:.1f}%", "ppo"),
    ("Precision Rate (>14)", f"{ppo_perfect_rate:.1f}%", f"{pg_perfect_rate:.1f}%", "ppo"),
]

# 헤더
y0 = 0.94
ax4.text(0.02, y0, "Metric", fontsize=11, fontweight="bold", va="center",
         transform=ax4.transAxes, color="#333333")
ax4.text(0.55, y0, "PPO", fontsize=12, fontweight="bold", va="center",
         transform=ax4.transAxes, color=PPO_COLOR, ha="center")
ax4.text(0.82, y0, "REINFORCE", fontsize=12, fontweight="bold", va="center",
         transform=ax4.transAxes, color=PG_COLOR, ha="center")

# 구분선
ax4.plot([0.02, 0.98], [0.90, 0.90], color="#CCCCCC", linewidth=1.5,
         transform=ax4.transAxes, clip_on=False)

# 행
row_y = 0.84
row_step = 0.087
for i, (label, ppo_val, pg_val, winner) in enumerate(metrics):
    y = row_y - i * row_step
    bg = "#F0F4F8" if i % 2 == 0 else "#FFFFFF"

    rect = FancyBboxPatch((0.01, y - 0.032), 0.98, 0.065,
                           boxstyle="round,pad=0.008",
                           facecolor=bg, edgecolor="none",
                           transform=ax4.transAxes)
    ax4.add_patch(rect)

    ax4.text(0.04, y, label, fontsize=10, va="center",
             transform=ax4.transAxes, color="#444444")

    # 승자 표시: 볼드 + 약간 큰 폰트
    ppo_fw = "bold" if winner == "ppo" else "normal"
    pg_fw = "bold" if winner == "pg" else "normal"
    ppo_fs = 10.5 if winner == "ppo" else 10
    pg_fs = 10.5 if winner == "pg" else 10

    ax4.text(0.55, y, ppo_val, fontsize=ppo_fs, va="center", fontweight=ppo_fw,
             transform=ax4.transAxes, color=PPO_COLOR, ha="center")
    ax4.text(0.82, y, pg_val, fontsize=pg_fs, va="center", fontweight=pg_fw,
             transform=ax4.transAxes, color=PG_COLOR, ha="center")

ax4.set_title("Performance Summary", fontsize=13, fontweight="bold", pad=10)

# ── 전체 타이틀 ──────────────────────────────────────────────────────────
fig.suptitle("PPO vs REINFORCE  |  2D Drone Drop (300K Timesteps, Fixed Target + Variable Speed)",
             fontsize=15, fontweight="bold", color="#222222", y=0.96)

save_path = os.path.join(RESULTS_DIR, "analysis_comparison.png")
plt.savefig(save_path, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close(fig)
print(f"\nSaved: {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 2: 누적 보상 비교 (타임스텝 기준)
# ════════════════════════════════════════════════════════════════════════════
fig2, ax = plt.subplots(figsize=(12, 5))
fig2.patch.set_facecolor("#FAFAFA")
ax.set_facecolor("#FFFFFF")

# 타임스텝 기준 누적 보상
# PPO: 14K 에피소드에서 각 에피소드당 ~22스텝 → 보상이 높음
# REINFORCE: 300K 에피소드에서 각 에피소드당 ~1스텝 → 보상이 낮음
# 공정 비교: 동일한 타임스텝 구간에서의 평균 보상 비교

# 타임스텝 구간별 평균 보상 (bin 방식)
n_bins = 100
bin_edges = np.linspace(0, TOTAL_TIMESTEPS, n_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

def binned_mean(timesteps, rewards, bin_edges):
    means = []
    for i in range(len(bin_edges) - 1):
        mask = (timesteps >= bin_edges[i]) & (timesteps < bin_edges[i+1])
        if mask.sum() > 0:
            means.append(np.mean(rewards[mask]))
        else:
            means.append(np.nan)
    return np.array(means)

ppo_binned = binned_mean(ppo_ts, ppo_rewards, bin_edges)
pg_binned = binned_mean(pg_ts, pg_rewards, bin_edges)

# 누적 평균 보상
ppo_cumavg = np.nancumsum(ppo_binned) / np.arange(1, len(ppo_binned) + 1)
pg_cumavg = np.nancumsum(pg_binned) / np.arange(1, len(pg_binned) + 1)

ax.plot(bin_centers, ppo_binned, color=PPO_COLOR, linewidth=2, label="PPO", alpha=0.9)
ax.plot(bin_centers, pg_binned, color=PG_COLOR, linewidth=2, label="REINFORCE", alpha=0.9)

ax.fill_between(bin_centers, ppo_binned, alpha=0.12, color=PPO_COLOR)
ax.fill_between(bin_centers, pg_binned, alpha=0.12, color=PG_COLOR)

ax.axhline(y=12.0, color=TARGET_COLOR, linestyle="--", alpha=0.5, linewidth=1.2, label="Target (12.0)")

ax.set_xlabel("Timestep", fontsize=12)
ax.set_ylabel("Mean Reward (per 3K-step bin)", fontsize=12)
ax.set_title("Average Reward Over Training (Timestep-aligned Bins)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.15)
ax.set_xlim(0, TOTAL_TIMESTEPS)
ax.set_ylim(-6, 16)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
plt.tight_layout()

save_path2 = os.path.join(RESULTS_DIR, "analysis_timestep_bins.png")
plt.savefig(save_path2, dpi=180, facecolor=fig2.get_facecolor(), edgecolor="none")
plt.close(fig2)
print(f"Saved: {save_path2}")

print("\nDone!")
