"""
evaluate.py — 통합 평가 스크립트

어떤 알고리즘이든 로드하여 평가 + 렌더링.

실행:
  cd "RL Workspace"
  python -m lunar_lander.evaluate --algo ppo_discrete --render
  python -m lunar_lander.evaluate --algo sac --render
  python -m lunar_lander.evaluate --all
"""

import argparse
import numpy as np
import gymnasium as gym
import torch

from .configs.base import BASE_CONFIG
from .configs.ppo_discrete import CONFIG as PPO_D_CFG
from .configs.ppo_continuous import CONFIG as PPO_C_CFG
from .configs.reinforce import CONFIG as RF_CFG
from .configs.dqn import CONFIG as DQN_CFG
from .configs.sac import CONFIG as SAC_CFG

# 알고리즘 레지스트리
ALGO_REGISTRY = {
    "ppo_discrete": PPO_D_CFG,
    "ppo_continuous": PPO_C_CFG,
    "reinforce": RF_CFG,
    "dqn": DQN_CFG,
    "sac": SAC_CFG,
}


def _make_agent(algo_name: str, config: dict, device: torch.device):
    """알고리즘 이름으로 에이전트를 생성하고 모델을 로드한다."""
    import os
    model_path = os.path.join(config["models_dir"], f"{algo_name}.pt")

    if algo_name == "ppo_discrete":
        from .agents.ppo_agent import PPOAgent
        agent = PPOAgent(
            obs_dim=config["obs_dim"], action_dim=config["action_dim"],
            hidden_dim=config["hidden_dim"], continuous=False, device=device,
        )
        agent.load(model_path)
        return agent, "ppo"

    elif algo_name == "ppo_continuous":
        from .agents.ppo_agent import PPOAgent
        agent = PPOAgent(
            obs_dim=config["obs_dim"], action_dim=config["action_dim"],
            hidden_dim=config["hidden_dim"], continuous=True, device=device,
        )
        agent.load(model_path)
        return agent, "ppo_cont"

    elif algo_name == "reinforce":
        from .agents.reinforce_agent import REINFORCEAgent
        agent = REINFORCEAgent(
            obs_dim=config["obs_dim"], action_dim=config["action_dim"],
            hidden_dim=config["hidden_dim"], device=device,
        )
        agent.load(model_path)
        return agent, "reinforce"

    elif algo_name == "dqn":
        from .agents.dqn_agent import DQNAgent
        agent = DQNAgent(
            obs_dim=config["obs_dim"], action_dim=config["action_dim"],
            hidden_dim=config["hidden_dim"], device=device,
        )
        agent.load(model_path)
        return agent, "dqn"

    elif algo_name == "sac":
        from .agents.sac_agent import SACAgent
        agent = SACAgent(
            obs_dim=config["obs_dim"], action_dim=config["action_dim"],
            hidden_dim=config["hidden_dim"], device=device,
        )
        agent.load(model_path)
        return agent, "sac"

    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def _get_action(agent, agent_type: str, obs):
    """평가용 deterministic 행동 선택."""
    if agent_type == "ppo":
        # Discrete: argmax (greedy)
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(agent.device)
            dist, _ = agent.network(obs_tensor)
            return dist.probs.argmax().item()
    elif agent_type == "ppo_cont":
        # Continuous: mean action (deterministic)
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(agent.device)
            dist, _ = agent.network(obs_tensor)
            return dist.mean.cpu().numpy()
    elif agent_type == "reinforce":
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(agent.device)
            dist = agent.policy(obs_tensor)
            return dist.probs.argmax().item()
    elif agent_type == "dqn":
        return agent.greedy_action(obs)
    elif agent_type == "sac":
        return agent.deterministic_action(obs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def evaluate_algo(algo_name: str, render: bool = False, episodes: int = 20):
    """단일 알고리즘 평가."""
    config = ALGO_REGISTRY[algo_name]
    device = torch.device("cpu")

    print(f"\n{'─'*50}")
    print(f"  {algo_name} 평가 — {config['env_id']}")
    print(f"{'─'*50}")

    try:
        agent, agent_type = _make_agent(algo_name, config, device)
    except FileNotFoundError:
        print(f"  모델 파일 없음 — 먼저 학습하세요.")
        return None

    render_mode = "human" if render else None
    env = gym.make(config["env_id"], render_mode=render_mode)

    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = _get_action(agent, agent_type, obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        if render:
            print(f"  에피소드 {ep+1}: {total_reward:.1f}")

    env.close()

    mean_r = np.mean(rewards)
    std_r = np.std(rewards)
    print(f"\n  결과 ({episodes} 에피소드):")
    print(f"    평균 보상: {mean_r:.1f} +/- {std_r:.1f}")
    print(f"    최고: {np.max(rewards):.1f}  |  최저: {np.min(rewards):.1f}")
    print(f"    해결 여부: {'O' if mean_r >= config['solved_threshold'] else 'X'} (기준: {config['solved_threshold']})")

    return {"algo": algo_name, "mean": mean_r, "std": std_r, "rewards": rewards}


def evaluate_all(episodes: int = 20):
    """모든 알고리즘 평가 + 요약 테이블."""
    results = []
    for algo_name in ALGO_REGISTRY:
        r = evaluate_algo(algo_name, render=False, episodes=episodes)
        if r is not None:
            results.append(r)

    if results:
        print(f"\n{'='*60}")
        print(f"  종합 비교 ({episodes} 에피소드)")
        print(f"{'='*60}")
        print(f"  {'알고리즘':<20s} {'평균 보상':>10s} {'표준편차':>10s} {'해결':>6s}")
        print(f"  {'─'*46}")
        for r in results:
            cfg = ALGO_REGISTRY[r["algo"]]
            solved = "O" if r["mean"] >= cfg["solved_threshold"] else "X"
            print(f"  {r['algo']:<20s} {r['mean']:>10.1f} {r['std']:>10.1f} {solved:>6s}")


def main():
    parser = argparse.ArgumentParser(description="LunarLander 알고리즘 평가")
    parser.add_argument("--algo", type=str, choices=list(ALGO_REGISTRY.keys()),
                        help="평가할 알고리즘")
    parser.add_argument("--render", action="store_true", help="Pygame 렌더링")
    parser.add_argument("--episodes", type=int, default=20, help="평가 에피소드 수")
    parser.add_argument("--all", action="store_true", help="모든 알고리즘 평가")

    args = parser.parse_args()

    if args.all:
        evaluate_all(episodes=args.episodes)
    elif args.algo:
        evaluate_algo(args.algo, render=args.render, episodes=args.episodes)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
