"""
evaluate.py - 학습된 PPO 모델 평가 및 시각화

실행:
  # 학습된 모델 평가 + pygame 렌더링
  python -m cartpole_ppo.evaluate

  # 미학습 vs 학습 비교 (렌더링 포함)
  python -m cartpole_ppo.evaluate --compare

  # 렌더링 없이 수치만 비교
  python -m cartpole_ppo.evaluate --compare --no-render
"""

import argparse
import os

import gymnasium as gym
import numpy as np
import torch

from .ppo_agent import PPOAgent
from .train import CONFIG


def _make_agent(model_path: str | None = None) -> PPOAgent:
    """
    PPOAgent를 생성한다.
    model_path가 None이면 랜덤 초기화 상태(미학습)로 반환한다.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(
        obs_dim=CONFIG["obs_dim"],
        action_dim=CONFIG["action_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        device=device,
    )
    if model_path is not None:
        agent.load(model_path)
    return agent


def _run_episodes(
    agent: PPOAgent,
    n_episodes: int,
    render: bool,
    label: str,
    save_gif: bool = False,
    gif_path: str = "results/cartpole_demo.gif",
) -> dict:
    """
    에이전트를 n_episodes 동안 실행하고 결과를 반환한다.

    Args:
        agent      : 평가할 PPOAgent
        n_episodes : 실행할 에피소드 수
        render     : pygame 창 렌더링 여부
        label      : 출력에 표시할 레이블 (예: "학습 완료", "미학습(랜덤)")
        save_gif   : 첫 에피소드를 GIF로 저장할지 여부
        gif_path   : GIF 저장 경로

    Returns:
        dict: episode_rewards, mean_reward, std_reward, success_rate
    """
    render_mode = "human" if render else "rgb_array"
    env = gym.make(CONFIG["env_id"], render_mode=render_mode)

    gif_frames = []
    episode_rewards = []

    print(f"\n{'='*60}")
    print(f"  [{label}]  ({n_episodes} 에피소드)")
    print(f"{'='*60}\n")

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0

        # GIF: 첫 에피소드만 녹화
        if save_gif and ep == 0:
            frame = env.render()
            if frame is not None:
                gif_frames.append(frame)

        while True:
            # Greedy 행동 선택 (평가 시에는 argmax 사용)
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(agent.device)
            with torch.no_grad():
                dist, _ = agent.network(obs_tensor)
                action = dist.probs.argmax().item()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            if save_gif and ep == 0:
                frame = env.render()
                if frame is not None:
                    gif_frames.append(frame)

            if done:
                break

        episode_rewards.append(ep_reward)
        success = "O" if ep_reward >= 500 else "X"
        print(f"  에피소드 {ep+1:2d}: {ep_reward:6.0f} 스텝  [{success}]")

    env.close()

    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    success_rate = float(np.mean([r >= 500 for r in episode_rewards]) * 100)

    print(f"\n  평균 보상 : {mean_reward:.1f} ± {std_reward:.1f}")
    print(f"  최대 보상 : {max(episode_rewards):.0f}")
    print(f"  성공률    : {success_rate:.0f}%")

    if save_gif and gif_frames:
        _save_gif(gif_frames, gif_path)

    return {
        "episode_rewards": episode_rewards,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "success_rate": success_rate,
    }


def evaluate(
    model_path: str,
    n_episodes: int = 10,
    render: bool = True,
    save_gif: bool = False,
    gif_path: str = "results/cartpole_demo.gif",
) -> dict:
    """학습된 모델을 로드해 평가한다."""
    if not os.path.exists(model_path):
        print(f"모델 파일을 찾을 수 없습니다: {model_path}")
        print("먼저 학습을 실행하세요: python -m cartpole_ppo.train")
        return {}

    agent = _make_agent(model_path)
    return _run_episodes(agent, n_episodes, render, label="학습 완료 모델", save_gif=save_gif, gif_path=gif_path)


def compare(
    model_path: str,
    n_episodes: int = 5,
    render: bool = True,
) -> None:
    """
    미학습 에이전트(랜덤 가중치)와 학습된 에이전트를 순서대로 실행해
    성능 차이를 비교한다.

    순서:
      1) 미학습 에이전트 실행 (pt 파일 미적용)
      2) 학습된 에이전트 실행 (pt 파일 적용)
      3) 비교 요약 출력
    """
    if not os.path.exists(model_path):
        print(f"모델 파일을 찾을 수 없습니다: {model_path}")
        return

    print(f"\n{'#'*60}")
    print(f"  미학습 vs 학습 완료 비교  ({n_episodes} 에피소드씩)")
    print(f"  모델: {model_path}")
    print(f"{'#'*60}")

    # ── ① 미학습 (랜덤 가중치) ───────────────────────────────────────────────
    untrained_agent = _make_agent(model_path=None)  # .pt 파일 로드 안 함
    untrained_result = _run_episodes(
        untrained_agent, n_episodes, render, label="미학습 (랜덤 가중치)"
    )

    # ── ② 학습 완료 ──────────────────────────────────────────────────────────
    trained_agent = _make_agent(model_path=model_path)
    trained_result = _run_episodes(
        trained_agent, n_episodes, render, label="학습 완료 모델"
    )

    # ── ③ 비교 요약 ──────────────────────────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"  비교 결과 요약")
    print(f"{'#'*60}")
    print(f"  {'':20s}  {'미학습':>10s}  {'학습 완료':>10s}")
    print(f"  {'─'*46}")
    print(f"  {'평균 보상':20s}  {untrained_result['mean_reward']:>10.1f}  {trained_result['mean_reward']:>10.1f}")
    print(f"  {'표준편차':20s}  {untrained_result['std_reward']:>10.1f}  {trained_result['std_reward']:>10.1f}")
    print(f"  {'성공률 (500스텝)':20s}  {untrained_result['success_rate']:>9.0f}%  {trained_result['success_rate']:>9.0f}%")

    improvement = trained_result["mean_reward"] - untrained_result["mean_reward"]
    print(f"\n  평균 보상 향상: +{improvement:.1f} 스텝")
    print(f"{'#'*60}\n")


def _save_gif(frames: list, path: str) -> None:
    """RGB 프레임 리스트를 GIF로 저장한다."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        img = ax.imshow(frames[0])

        def update(frame_idx):
            img.set_array(frames[frame_idx])
            return [img]

        ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=20, blit=True)
        ani.save(path, writer="pillow", fps=50)
        plt.close(fig)
        print(f"GIF 저장 완료: {path}")
    except ImportError:
        print("GIF 저장 실패: pillow 패키지가 필요합니다. (pip install pillow)")
    except Exception as e:
        print(f"GIF 저장 실패: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO CartPole 평가")
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(CONFIG["results_dir"], CONFIG["model_filename"]),
        help="모델 파일 경로 (기본: results/ppo_cartpole.pt)",
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="평가 에피소드 수 (기본: 10)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="미학습 vs 학습 완료 비교 모드",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="렌더링 없이 수치만 출력",
    )
    parser.add_argument(
        "--save-gif", action="store_true",
        help="첫 에피소드를 GIF로 저장 (--compare 와 함께 사용 불가)",
    )
    args = parser.parse_args()

    if args.compare:
        compare(
            model_path=args.model,
            n_episodes=args.episodes,
            render=not args.no_render,
        )
    else:
        evaluate(
            model_path=args.model,
            n_episodes=args.episodes,
            render=not args.no_render,
            save_gif=args.save_gif,
        )
