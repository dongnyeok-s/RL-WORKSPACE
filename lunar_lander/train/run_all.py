"""
run_all.py — 5개 알고리즘 순차 학습

실행:
  cd lunar_lander
  python -m train.run_all
"""

import time


def main():
    print("=" * 60)
    print("  LunarLander — 5개 알고리즘 순차 학습")
    print("=" * 60)

    total_start = time.time()

    # 1. PPO Discrete
    print("\n[1/5] PPO Discrete")
    from .train_ppo import train as train_ppo
    train_ppo(continuous=False)

    # 2. REINFORCE
    print("\n[2/5] REINFORCE")
    from .train_reinforce import train as train_reinforce
    train_reinforce()

    # 3. DQN
    print("\n[3/5] Double DQN")
    from .train_dqn import train as train_dqn
    train_dqn()

    # 4. PPO Continuous
    print("\n[4/5] PPO Continuous")
    train_ppo(continuous=True)

    # 5. SAC
    print("\n[5/5] SAC")
    from .train_sac import train as train_sac
    train_sac()

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  전체 학습 완료! 총 소요 시간: {total_elapsed/60:.1f}분")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
