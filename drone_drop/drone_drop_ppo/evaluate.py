"""
evaluate.py - 학습된 PPO 모델 평가 및 시각화

실행:
  # 학습된 모델 평가 + pygame 렌더링
  python -m drone_drop_ppo.evaluate

  # 미학습 vs 학습 비교 (렌더링 포함)
  python -m drone_drop_ppo.evaluate --compare

  # 렌더링 없이 수치만 비교
  python -m drone_drop_ppo.evaluate --compare --no-render
"""

import argparse
import os

import numpy as np
import torch

from .ppo_agent import PPOAgent
from .pg_agent import REINFORCEAgent
from .env import DroneDropEnv, DRONE_ALTITUDE, GRAVITY, WORLD_WIDTH, WORLD_HEIGHT, GROUND_HEIGHT, TARGET_X, DT
from .train import CONFIG


def _animate_drop(env, info, render_mode="human", gif_frames=None):
    """
    투하 후 포물선 낙하를 애니메이션으로 보여준다.
    env.step()은 즉시 종료하므로, 렌더링용으로만 사용.
    """
    try:
        import pygame
        import time
    except ImportError:
        return

    drop_x = info["drop_x"]
    speed = info["speed"]

    # 패키지 물리 시뮬레이션
    pkg_x = drop_x
    pkg_y = DRONE_ALTITUDE
    pkg_vx = speed
    pkg_vy = 0.0
    drone_x = drop_x  # 드론도 계속 이동

    while pkg_y > 0:
        pkg_vy += GRAVITY * DT
        pkg_x += pkg_vx * DT
        pkg_y -= pkg_vy * DT
        drone_x += speed * DT

        if pkg_y < 0:
            pkg_y = 0

        # 렌더링: env의 상태를 임시로 업데이트
        env.drone_x = drone_x
        env.pkg_x = pkg_x
        env.pkg_y = pkg_y
        env.pkg_landed = (pkg_y <= 0)
        env.render()

        if gif_frames is not None:
            frame = env.render()
            if frame is not None:
                gif_frames.append(frame)

    # 착지 후 잠시 대기
    for _ in range(30):  # 1초
        env.render()
        if gif_frames is not None:
            frame = env.render()
            if frame is not None:
                gif_frames.append(frame)


def _make_agent(model_path: str | None = None) -> PPOAgent:
    """PPOAgent를 생성한다. model_path=None이면 미학습 상태."""
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
    gif_path: str = "results/drone_drop_demo.gif",
) -> dict:
    """
    에이전트를 n_episodes 동안 실행하고 결과를 반환한다.

    Returns:
        dict: episode_rewards, distances, mean_reward, mean_distance, success_rate
    """
    render_mode = "human" if render else "rgb_array"
    env = DroneDropEnv(render_mode=render_mode)

    gif_frames = []
    episode_rewards = []
    distances = []

    print(f"\n{'='*60}")
    print(f"  [{label}]  ({n_episodes} 에피소드)")
    print(f"{'='*60}\n")

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0.0

        if save_gif and ep == 0:
            frame = env.render()
            if frame is not None:
                gif_frames.append(frame)

        while True:
            # Greedy 행동 선택 (평가 시 argmax)
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(agent.device)
            with torch.no_grad():
                dist, _ = agent.network(obs_tensor)
                action = dist.probs.argmax().item()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            if render:
                env.render()

            if save_gif and ep == 0:
                frame = env.render()
                if frame is not None:
                    gif_frames.append(frame)

            if done:
                # 투하 후 포물선 낙하 애니메이션 (렌더링 모드에서만)
                if terminated and render and info.get("drop_x") is not None:
                    _animate_drop(env, info, render_mode="human" if render else "rgb_array",
                                  gif_frames=gif_frames if (save_gif and ep == 0) else None)
                break

        episode_rewards.append(ep_reward)
        distance = info.get("distance")
        speed = info.get("speed", 0)
        if distance is not None:
            distances.append(distance)
            success = "O" if distance < 30.0 else "X"
            print(f"  에피소드 {ep+1:2d}: 보상 {ep_reward:7.2f}  |  속도: {speed:5.0f}  |  착지 거리: {distance:6.1f}px  [{success}]")
        else:
            print(f"  에피소드 {ep+1:2d}: 보상 {ep_reward:7.2f}  |  속도: {speed:5.0f}  |  미투하 [X]")

    env.close()

    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    mean_distance = float(np.mean(distances)) if distances else float("inf")
    success_rate = float(np.mean([d < 30.0 for d in distances]) * 100) if distances else 0.0

    print(f"\n  평균 보상    : {mean_reward:.2f} ± {std_reward:.2f}")
    if distances:
        print(f"  평균 착지 거리: {mean_distance:.1f}px")
        print(f"  성공률 (<30px): {success_rate:.0f}%")

    if save_gif and gif_frames:
        _save_gif(gif_frames, gif_path)

    return {
        "episode_rewards": episode_rewards,
        "distances": distances,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_distance": mean_distance,
        "success_rate": success_rate,
    }


def evaluate(
    model_path: str,
    n_episodes: int = 10,
    render: bool = True,
    save_gif: bool = False,
    gif_path: str = "results/drone_drop_demo.gif",
) -> dict:
    """학습된 모델을 로드해 평가한다."""
    if not os.path.exists(model_path):
        print(f"모델 파일을 찾을 수 없습니다: {model_path}")
        print("먼저 학습을 실행하세요: python -m drone_drop_ppo.train")
        return {}

    agent = _make_agent(model_path)
    return _run_episodes(agent, n_episodes, render, label="학습 완료 모델", save_gif=save_gif, gif_path=gif_path)


def compare(
    model_path: str,
    n_episodes: int = 5,
    render: bool = True,
) -> None:
    """미학습 에이전트와 학습된 에이전트의 성능을 비교한다."""
    if not os.path.exists(model_path):
        print(f"모델 파일을 찾을 수 없습니다: {model_path}")
        return

    print(f"\n{'#'*60}")
    print(f"  미학습 vs 학습 완료 비교  ({n_episodes} 에피소드씩)")
    print(f"  모델: {model_path}")
    print(f"{'#'*60}")

    # ── ① 미학습 (랜덤 가중치) ───────────────────────────────────────────────
    untrained_agent = _make_agent(model_path=None)
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
    print(f"  {'평균 보상':20s}  {untrained_result['mean_reward']:>10.2f}  {trained_result['mean_reward']:>10.2f}")
    print(f"  {'평균 착지 거리':20s}  {untrained_result['mean_distance']:>9.1f}px  {trained_result['mean_distance']:>9.1f}px")
    print(f"  {'성공률 (<30px)':20s}  {untrained_result['success_rate']:>9.0f}%  {trained_result['success_rate']:>9.0f}%")

    improvement = trained_result["mean_reward"] - untrained_result["mean_reward"]
    print(f"\n  평균 보상 향상: +{improvement:.2f}")
    print(f"{'#'*60}\n")


def _make_pg_agent(model_path: str | None = None) -> REINFORCEAgent:
    """REINFORCEAgent를 생성한다. model_path=None이면 미학습 상태."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = REINFORCEAgent(
        obs_dim=CONFIG["obs_dim"],
        action_dim=CONFIG["action_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        device=device,
    )
    if model_path is not None:
        agent.load(model_path)
    return agent


def _run_pg_episodes(
    agent: REINFORCEAgent,
    n_episodes: int,
    render: bool,
    label: str,
) -> dict:
    """REINFORCE 에이전트를 n_episodes 동안 실행하고 결과를 반환한다."""
    render_mode = "human" if render else "rgb_array"
    env = DroneDropEnv(render_mode=render_mode)

    episode_rewards = []
    distances = []

    print(f"\n{'='*60}")
    print(f"  [{label}]  ({n_episodes} 에피소드)")
    print(f"{'='*60}\n")

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0.0

        while True:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(agent.device)
            with torch.no_grad():
                dist = agent.policy(obs_tensor)
                action = dist.probs.argmax().item()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            if render:
                env.render()

            if done:
                if terminated and render and info.get("drop_x") is not None:
                    _animate_drop(env, info, render_mode="human" if render else "rgb_array")
                break

        episode_rewards.append(ep_reward)
        distance = info.get("distance")
        speed = info.get("speed", 0)
        if distance is not None:
            distances.append(distance)
            success = "O" if distance < 30.0 else "X"
            print(f"  에피소드 {ep+1:2d}: 보상 {ep_reward:7.2f}  |  속도: {speed:5.0f}  |  착지 거리: {distance:6.1f}px  [{success}]")
        else:
            print(f"  에피소드 {ep+1:2d}: 보상 {ep_reward:7.2f}  |  속도: {speed:5.0f}  |  미투하 [X]")

    env.close()

    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))
    mean_distance = float(np.mean(distances)) if distances else float("inf")
    success_rate = float(np.mean([d < 30.0 for d in distances]) * 100) if distances else 0.0

    print(f"\n  평균 보상    : {mean_reward:.2f} ± {std_reward:.2f}")
    if distances:
        print(f"  평균 착지 거리: {mean_distance:.1f}px")
        print(f"  성공률 (<30px): {success_rate:.0f}%")

    return {
        "episode_rewards": episode_rewards,
        "distances": distances,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_distance": mean_distance,
        "success_rate": success_rate,
    }


def compare_pg(
    ppo_model_path: str,
    pg_model_path: str,
    n_episodes: int = 5,
    render: bool = True,
) -> None:
    """REINFORCE와 PPO의 성능을 Pygame으로 비교한다."""
    print(f"\n{'#'*60}")
    print(f"  REINFORCE vs PPO 비교  ({n_episodes} 에피소드씩)")
    print(f"{'#'*60}")

    # ── ① REINFORCE ──────────────────────────────────────────────────────────
    pg_agent = _make_pg_agent(pg_model_path if os.path.exists(pg_model_path) else None)
    pg_result = _run_pg_episodes(pg_agent, n_episodes, render, label="REINFORCE (Vanilla PG)")

    # ── ② PPO ─────────────────────────────────────────────────────────────────
    ppo_agent = _make_agent(ppo_model_path if os.path.exists(ppo_model_path) else None)
    ppo_result = _run_episodes(ppo_agent, n_episodes, render, label="PPO")

    # ── ③ 비교 요약 ──────────────────────────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"  비교 결과 요약")
    print(f"{'#'*60}")
    print(f"  {'':20s}  {'REINFORCE':>12s}  {'PPO':>12s}")
    print(f"  {'─'*50}")
    print(f"  {'평균 보상':20s}  {pg_result['mean_reward']:>12.2f}  {ppo_result['mean_reward']:>12.2f}")
    print(f"  {'평균 착지 거리':20s}  {pg_result['mean_distance']:>11.1f}px  {ppo_result['mean_distance']:>11.1f}px")
    print(f"  {'성공률 (<30px)':20s}  {pg_result['success_rate']:>11.0f}%  {ppo_result['success_rate']:>11.0f}%")
    print(f"{'#'*60}\n")


def _save_gif(frames: list, path: str) -> None:
    """RGB 프레임 리스트를 GIF로 저장한다."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis("off")
        img = ax.imshow(frames[0])

        def update(frame_idx):
            img.set_array(frames[frame_idx])
            return [img]

        ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=33, blit=True)
        ani.save(path, writer="pillow", fps=30)
        plt.close(fig)
        print(f"GIF 저장 완료: {path}")
    except ImportError:
        print("GIF 저장 실패: pillow 패키지가 필요합니다. (pip install pillow)")
    except Exception as e:
        print(f"GIF 저장 실패: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Drone Drop 평가")
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(CONFIG["results_dir"], CONFIG["model_filename"]),
        help="모델 파일 경로 (기본: results/ppo_drone_drop.pt)",
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
        help="첫 에피소드를 GIF로 저장",
    )
    parser.add_argument(
        "--compare-pg", action="store_true",
        help="REINFORCE vs PPO 비교 모드",
    )
    parser.add_argument(
        "--pg-model",
        type=str,
        default=os.path.join(CONFIG["results_dir"], "pg_drone_drop.pt"),
        help="REINFORCE 모델 파일 경로 (기본: results/pg_drone_drop.pt)",
    )
    args = parser.parse_args()

    if args.compare_pg:
        compare_pg(
            ppo_model_path=args.model,
            pg_model_path=args.pg_model,
            n_episodes=args.episodes,
            render=not args.no_render,
        )
    elif args.compare:
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
