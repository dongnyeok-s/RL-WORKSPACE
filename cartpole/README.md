# CartPole — PPO vs REINFORCE 비교 실험

Gymnasium 내장 환경 CartPole-v1에서 PPO와 REINFORCE를 구현하고 비교한다.

---

## 환경

```
          ┌─────┐
          │  │  │ ← 폴 (쓰러지면 종료)
          └──┼──┘
        ─────┴─────  ← 카트 (좌/우 이동)
   ◄────────────────────►
```

| 항목 | 값 |
|------|-----|
| 관측 | 카트 위치, 카트 속도, 폴 각도, 폴 각속도 (4차원) |
| 행동 | 왼쪽(0), 오른쪽(1) (Discrete 2) |
| 보상 | 매 스텝 +1 (폴이 서 있는 한) |
| 종료 | 폴 각도 > 12도, 카트 위치 > 2.4, 또는 500스텝 |
| 만점 | 500 (500스텝 동안 폴을 세우면 성공) |

---

## 파일 구조

```
cartpole/
├── README.md
├── docs/
│   ├── PPO_개념정리.md              RL/PPO 이론 정리 문서
│   └── REINFORCE_vs_PPO_비교분석.md  실험 결과 비교 문서
├── cartpole_ppo/
│   ├── network.py      Actor-Critic 신경망 (Backbone + Actor/Critic Head)
│   ├── buffer.py        Rollout Buffer + GAE 계산
│   ├── ppo_agent.py     PPO Clip Loss 업데이트 로직
│   ├── pg_agent.py      REINFORCE 에이전트 (PPO 비교용)
│   ├── train.py         PPO 학습 스크립트
│   ├── train_pg.py      REINFORCE 학습 스크립트
│   └── evaluate.py      모델 평가 + 렌더링
└── results/
    ├── ppo_cartpole.pt          학습된 PPO 모델 가중치
    ├── ppo_ep_rewards.npy       PPO 에피소드 보상 기록
    ├── pg_ep_rewards.npy        REINFORCE 에피소드 보상 기록
    ├── training_curve.png       PPO 학습 곡선
    ├── pg_training_curve.png    REINFORCE 학습 곡선
    └── comparison.png           PPO vs REINFORCE 비교 차트
```

---

## 실행 방법

```bash
# 이 디렉토리에서 실행
cd cartpole

# PPO 학습 (500K steps, ~2분)
python -m cartpole_ppo.train

# REINFORCE 학습 (500K steps, PPO 결과 있으면 비교 그래프 자동 생성)
python -m cartpole_ppo.train_pg

# 학습된 모델 평가 + 렌더링
python -m cartpole_ppo.evaluate

# 미학습 vs 학습 비교 (렌더링 포함)
python -m cartpole_ppo.evaluate --compare

# 렌더링 없이 수치만 비교
python -m cartpole_ppo.evaluate --compare --no-render
```

---

## 코드 읽는 순서

PPO를 처음 공부한다면 이 순서를 권장:

```
1. network.py   — 신경망 구조 (Actor + Critic = 하나의 네트워크)
2. buffer.py    — 데이터 수집 + GAE 어드밴티지 계산
3. ppo_agent.py — PPO Clip Loss + 업데이트 로직
4. train.py     — 전체 학습 루프 (위 3개를 엮는 메인 코드)
5. pg_agent.py  — REINFORCE와 비교 (PPO가 왜 좋은지 체감)
```

---

## 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| total_timesteps | 500,000 | 총 학습 스텝 |
| rollout_length | 2,048 | 한 rollout 수집 스텝 |
| n_epochs | 10 | rollout당 업데이트 반복 |
| batch_size | 64 | 미니배치 크기 |
| lr | 3e-4 | Adam 학습률 |
| gamma | 0.99 | 할인율 |
| lam | 0.95 | GAE lambda |
| clip_eps | 0.2 | PPO clip 범위 |
| hidden_dim | 64 | 은닉층 크기 |

---

## 실험 결과

| | REINFORCE | PPO |
|---|---|---|
| 최종 평균 보상 | 500.0 (만점) | 500.0 (만점) |
| 수렴 속도 | ~300K steps | ~80K steps |
| 학습 안정성 | 불안정 (보상 등락 심함) | 안정적 수렴 |

> CartPole은 단순한 환경이라 두 알고리즘 모두 만점에 도달한다.
> PPO의 진가는 더 복잡한 환경 ([drone_drop](../drone_drop/))에서 나타난다.
