# LunarLander — 4대 RL 알고리즘 비교 실험

Gymnasium 내장 LunarLander 환경에서 **PPO, DQN, SAC, REINFORCE** 4개 알고리즘을 구현하고 비교한다.

이산(Discrete)과 연속(Continuous) 행동 공간을 모두 다루어 On-Policy vs Off-Policy, Value-based vs Policy-based의 차이를 실험적으로 확인한다.

---

## 환경

```
           ╲     ╱
            ╲   ╱
             ╲ ╱     ← 달 착륙선
            ┌─┴─┐
          🔥│   │🔥   ← 좌우 엔진
            └───┘
               🔥     ← 메인 엔진
    ─ ─ ─ ─ 🏁 ─ ─ ─ ─  ← 착지 패드
```

| 항목 | Discrete | Continuous |
|------|----------|------------|
| 관측 | 8차원 (x, y, vx, vy, angle, angular_vel, left_leg, right_leg) | 동일 |
| 행동 | 4개 (noop, left, main, right) | 2차원 (main [-1,1], lateral [-1,1]) |
| 보상 | 착지 +100~140, 추락 -100, 연료 소비 | 동일 |
| 해결 | 평균 보상 >= 200 | 동일 |

---

## 알고리즘

| 알고리즘 | 타입 | 환경 | 핵심 특징 |
|---------|------|------|----------|
| **PPO** | On-Policy, Actor-Critic | Discrete + Continuous | Clip Loss, GAE, Multiple Epochs |
| **DQN** | Off-Policy, Value-based | Discrete | Double DQN, Target Net, ε-greedy |
| **SAC** | Off-Policy, Actor-Critic | Continuous | Twin Q, Entropy Max, 자동 α |
| **REINFORCE** | On-Policy, Policy Gradient | Discrete | Baseline 비교용 (Vanilla PG) |

---

## 파일 구조

```
lunar_lander/
├── README.md
├── configs/
│   ├── base.py                 공유 설정
│   ├── ppo_discrete.py
│   ├── ppo_continuous.py
│   ├── reinforce.py
│   ├── dqn.py
│   └── sac.py
│
├── networks/
│   ├── actor_critic_discrete.py    Categorical ActorCritic (PPO 이산)
│   ├── actor_critic_continuous.py  Gaussian ActorCritic (PPO 연속)
│   ├── policy_net.py               PolicyNet (REINFORCE)
│   ├── q_network.py                QNetwork (DQN)
│   └── sac_networks.py             TwinQ + GaussianPolicy (SAC)
│
├── buffers/
│   ├── rollout_buffer.py           On-Policy + GAE (PPO용)
│   └── replay_buffer.py            Off-Policy (DQN/SAC용)
│
├── agents/
│   ├── ppo_agent.py                PPO (discrete/continuous 모드)
│   ├── reinforce_agent.py          REINFORCE
│   ├── dqn_agent.py                Double DQN
│   └── sac_agent.py                SAC
│
├── train/
│   ├── train_ppo.py                PPO (--continuous 플래그)
│   ├── train_reinforce.py
│   ├── train_dqn.py
│   ├── train_sac.py
│   └── run_all.py                  5개 변형 순차 실행
│
├── evaluate.py                     통합 평가 + 렌더링
├── compare.py                      비교 분석 차트 생성
└── results/
    ├── models/{algo}.pt
    ├── plots/{algo}.png
    └── tensorboard/{algo}/
```

---

## 실행 방법

```bash
# RL Workspace 루트에서 실행
cd "RL Workspace"

# 의존성 설치
brew install swig
pip install gymnasium[box2d]

# ── 개별 학습 ──────────────────────────────────────────
python -m lunar_lander.train.train_ppo                # PPO Discrete
python -m lunar_lander.train.train_ppo --continuous   # PPO Continuous
python -m lunar_lander.train.train_reinforce          # REINFORCE
python -m lunar_lander.train.train_dqn                # DQN
python -m lunar_lander.train.train_sac                # SAC

# ── 전체 순차 학습 ─────────────────────────────────────
python -m lunar_lander.train.run_all

# ── 평가 ───────────────────────────────────────────────
python -m lunar_lander.evaluate --algo ppo_discrete --render   # 렌더링
python -m lunar_lander.evaluate --algo sac --render
python -m lunar_lander.evaluate --all                          # 전체 수치 평가

# ── 비교 분석 ──────────────────────────────────────────
python -m lunar_lander.compare                         # 비교 차트 생성

# ── TensorBoard ────────────────────────────────────────
tensorboard --logdir lunar_lander/results/tensorboard
```

---

## 코드 읽는 순서

```
1. configs/base.py          — 공유 설정 확인
2. networks/ 중 하나        — 신경망 구조 이해
3. buffers/                 — On-Policy vs Off-Policy 버퍼 차이
4. agents/ 중 하나          — 알고리즘 업데이트 로직
5. train/ 중 하나           — 전체 학습 루프
6. evaluate.py + compare.py — 평가 및 비교
```

---

## 하이퍼파라미터

| 파라미터 | PPO | PPO-Cont | REINFORCE | DQN | SAC |
|---------|-----|----------|-----------|-----|-----|
| hidden_dim | 64 | 64 | 64 | 64 | 256 |
| lr | 3e-4 | 3e-4 | 1e-3 | 1e-3 | 3e-4 |
| gamma | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| batch_size | 64 | 64 | - | 64 | 256 |
| buffer/rollout | 2048 | 2048 | - | 100K | 100K |
| 고유 설정 | clip=0.2, 10epochs | entropy=0 | - | ε: 1→0.05 | α: auto |

총 학습 스텝: 500K (모든 알고리즘 동일)

---

## 핵심 인사이트

| 비교 축 | 결과 |
|---------|------|
| **PPO vs REINFORCE** | PPO가 GAE + Clip으로 안정적 수렴. REINFORCE는 고분산 |
| **PPO vs DQN** | 이산 환경에서 비슷한 최종 성능. DQN이 샘플 효율적 |
| **SAC vs PPO-Cont** | SAC가 연속 환경에서 더 빠르게 수렴 (Off-Policy 효율 + 엔트로피 탐험) |
| **On vs Off-Policy** | Off-Policy (DQN, SAC)가 데이터 재사용으로 샘플 효율적 |
