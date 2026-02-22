# REINFORCE vs PPO 비교분석 — CartPole-v1

동일 조건(500K timesteps, seed=42)에서 두 알고리즘을 학습시킨 결과.

---

## 1. 실험 조건

| 항목 | REINFORCE | PPO |
|------|-----------|-----|
| 환경 | CartPole-v1 | CartPole-v1 |
| 총 타임스텝 | 500,000 | 500,000 |
| 신경망 구조 | MLP 64-64 (Actor only) | MLP 64-64 (Actor-Critic) |
| 학습률 | 3e-4 | 3e-4 |
| 할인율 (gamma) | 0.99 | 0.99 |
| 업데이트 방식 | 에피소드 완주 → 1회 업데이트 | 2048 스텝 rollout → 10 epoch 업데이트 |
| Clip | 없음 | epsilon=0.2 |
| Entropy Bonus | 없음 | coef=0.01 |
| GAE | 없음 (Monte Carlo G_t) | lambda=0.95 |

---

## 2. 학습 결과 요약

| 지표 | REINFORCE | PPO |
|------|-----------|-----|
| 총 에피소드 수 | 1,405 | 2,970 |
| 평균 에피소드 길이 | 356 스텝 | 169 스텝 |
| 후반부 평균 보상 (250K~) | 475 | 200 |
| 후반부 표준편차 (250K~) | 66 | 172 |
| 학습 중 500 달성률 | 51.8% | 10.3% |

---

## 3. 평가 결과

학습 완료 후 동일 모델로 20 에피소드 평가.

| 평가 방식 | REINFORCE | PPO |
|-----------|-----------|-----|
| **Greedy** (argmax) | 500.0 (20/20 성공) | 500.0 (20/20 성공) |
| **Stochastic** (sample) | 493.5 (19/20 성공) | 295.4 (5/20 성공) |

**두 알고리즘 모두 greedy 평가에서 만점** 달성.

---

## 4. 핵심 발견

### 4-1. PPO 학습 곡선이 더 낮고 불안정한 이유

PPO의 entropy bonus가 정책을 의도적으로 불확실하게 유지한다:

```
REINFORCE 정책 출력:  Left=0.730  Right=0.270  (확신 높음)
PPO 정책 출력:       Left=0.573  Right=0.427  (의도적 불확실)
```

- REINFORCE: 확신이 높아서 학습 중 stochastic 행동도 대부분 정답 → 높은 보상
- PPO: entropy bonus로 탐험을 유지해서 stochastic 행동이 자주 오답 → 낮은 보상

CartPole은 **한 번의 잘못된 행동이 치명적**인 환경이므로, 이 확률 차이가 큰 성능 차이로 이어진다.

```
매 스텝 정답 확률 90%일 때, 500스텝 전부 정답일 확률:
0.9^500 ≈ 0 (사실상 불가능)

→ stochastic 행동 선택 시 에피소드 보상이 낮은 건 정책이 나쁜 게 아니라 탐험 비용
```

### 4-2. PPO 에피소드 수가 2배인 이유

구조적 차이:
- PPO: 초반 랜덤 정책에서 짧은 에피소드(평균 91스텝)가 대량 발생
- REINFORCE: 에피소드 단위 업데이트라 빠르게 학습 → 초반부터 에피소드가 비교적 김

PPO의 2048 스텝 rollout 안에 여러 개의 짧은 에피소드가 포함되므로 총 에피소드 수가 많아진다.

### 4-3. CartPole에서 REINFORCE가 잘 작동하는 이유

CartPole은 단순한 환경이라:
- 관측 4차원, 행동 2개 (이산)
- 최대 500스텝으로 짧음
- 보상 구조가 단순 (+1/스텝)

이런 환경에서는 REINFORCE의 단순함이 오히려 장점:
- Critic이 없어도 G_t 추정이 충분히 정확
- Clip이 없어도 정책이 크게 무너지지 않음
- Entropy bonus가 없어서 빠르게 확신 있는 정책에 수렴

---

## 5. 그렇다면 PPO는 왜 필요한가?

PPO의 장점은 **더 복잡한 환경**에서 나타난다:

| 특성 | 단순 환경 (CartPole) | 복잡한 환경 (로봇, 드론 등) |
|------|---------------------|--------------------------|
| 관측 차원 | 4 | 수십~수백 |
| 행동 공간 | 이산 2개 | 연속 다차원 |
| 에피소드 길이 | ~500 | 수천~수만 |
| 보상 구조 | 단순 | 희소/복잡 |
| REINFORCE 성능 | 충분 | G_t 분산 폭발, 학습 불가 |
| PPO 필요성 | 낮음 | **높음** |

PPO의 핵심 메커니즘이 빛나는 상황:
- **GAE**: 긴 에피소드에서 G_t 분산을 줄여줌
- **Clip**: 고차원 행동 공간에서 치명적 업데이트 방지
- **Multiple Epochs**: 데이터 수집이 비용이 큰 환경에서 효율성 향상

---

## 6. 비교 차트

![REINFORCE vs PPO 비교 차트](results/comparison.png)

- **상단**: 타임스텝 기준 학습 곡선 (±1 std 밴드)
- **중단 왼쪽**: 롤링 변동성 (50K 윈도우)
- **중단 오른쪽**: 에피소드 보상 분포
- **하단**: 정량 비교 테이블

---

## 7. 코드 위치

| 파일 | 설명 |
|------|------|
| `cartpole_ppo/pg_agent.py` | REINFORCE 에이전트 |
| `cartpole_ppo/train_pg.py` | REINFORCE 학습 루프 |
| `cartpole_ppo/ppo_agent.py` | PPO 에이전트 |
| `cartpole_ppo/train.py` | PPO 학습 루프 |
| `cartpole_ppo/evaluate.py` | 모델 평가 + 비교 |

---

## 실행 방법

```bash
cd cartpole/

# PPO 학습
python -m cartpole_ppo.train

# REINFORCE 학습 (PPO 학습 후 비교 차트 자동 생성)
python -m cartpole_ppo.train_pg

# 학습된 PPO 모델 평가
python -m cartpole_ppo.evaluate

# 미학습 vs 학습 비교
python -m cartpole_ppo.evaluate --compare --no-render
```
