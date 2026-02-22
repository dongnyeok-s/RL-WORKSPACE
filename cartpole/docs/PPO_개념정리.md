# PPO 개념 정리

## 목차

1. [강화학습 기본 개념](#1-강화학습-기본-개념)
2. [Policy Gradient (REINFORCE)](#2-policy-gradient-reinforce)
3. [Policy Gradient의 문제점](#3-policy-gradient의-문제점)
4. [PPO가 문제를 해결하는 방법](#4-ppo가-문제를-해결하는-방법)
5. [PPO 핵심 개념 5가지](#5-ppo-핵심-개념-5가지)
6. [코드 위치 요약](#6-코드-위치-요약)
7. [드론 확장 포인트](#7-드론-확장-포인트)

---

## 1. 강화학습 기본 개념

강화학습은 **에이전트(agent)** 가 **환경(environment)** 과 상호작용하며 보상(reward)을 최대화하도록 스스로 학습하는 방식이다.

```
에이전트                       환경
   │                            │
   │──── 행동 a_t ────────────>│
   │                            │
   │<─── 관측 s_{t+1} ─────────│
   │<─── 보상 r_t ─────────────│
   │                            │
```

| 용어 | 의미 | CartPole 예시 |
|------|------|--------------|
| 상태 s | 에이전트가 관측하는 환경 정보 | (카트 위치, 속도, 폴 각도, 각속도) |
| 행동 a | 에이전트가 선택하는 동작 | 왼쪽(0) 또는 오른쪽(1) |
| 보상 r | 행동에 대한 환경의 피드백 | 폴이 쓰러지지 않으면 +1 |
| 정책 π | 상태 → 행동 확률의 매핑 | 신경망 |
| 에피소드 | 시작~종료까지 한 번의 시도 | 폴이 쓰러질 때까지 |

---

## 2. Policy Gradient (REINFORCE)

PPO의 출발점. 가장 기본적인 정책 학습 알고리즘이다.

### 핵심 아이디어

> "보상을 많이 받은 행동의 확률을 높인다."

수식:
```
∇J(θ) = E_t [ ∇ log π_θ(a_t|s_t) · G_t ]
```

- `log π_θ(a_t|s_t)` : 이 상황에서 이 행동을 선택한 확률의 log
- `G_t` : 이 시점 이후 받은 누적 보상 (Monte Carlo return)
  ```
  G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
  ```

### 학습 사이클

```
① 에피소드 완주 (막대 쓰러질 때까지)
② G_t 계산 (끝에서부터 역방향)
③ 신경망 1번 업데이트
④ 처음부터 반복
```

### 코드 위치

- `pg_agent.py` — REINFORCEAgent 구현
- `train_pg.py` — 에피소드 기반 학습 루프

---

## 3. Policy Gradient의 문제점

### 문제 1: 높은 분산 (High Variance)

G_t는 에피소드 전체 보상의 합이므로 에피소드마다 값이 크게 달라진다.

```
에피소드 1: G_0 = 12  → 신경망 조금 업데이트
에피소드 2: G_0 = 87  → 신경망 많이 업데이트
에피소드 3: G_0 = 23  → 신경망 조금 업데이트
```

→ 학습 곡선이 위아래로 크게 흔들리며 불안정하다.

### 문제 2: 치명적 업데이트 (Catastrophic Update)

G_t가 우연히 매우 큰 값이 되면 신경망이 너무 크게 업데이트된다.

```
어렵게 평균 보상 200까지 올렸는데
한 번의 잘못된 업데이트로 평균 20으로 무너질 수 있다.
```

학습 곡선에서는 이렇게 보인다:
```
보상
300 │              ╭─╮
200 │         ╭───╯  ╰╮
100 │    ╭────╯        ╰─────  ← 무너짐
 0  │────╯
    └─────────────────────────> 에피소드
```

### 문제 3: 데이터 낭비

에피소드 하나를 학습에 딱 한 번만 사용하고 버린다.
같은 데이터를 반복 학습하면 이미 크게 변한 정책과 데이터가 맞지 않아 오히려 해가 된다.

---

## 4. PPO가 문제를 해결하는 방법

| 문제 | PPO의 해결책 |
|------|-------------|
| 높은 분산 | GAE(λ=0.95)로 단기/장기 보상 균형 조절 |
| 치명적 업데이트 | Clip으로 업데이트 크기를 ε=0.2로 제한 |
| 데이터 낭비 | Clip 덕분에 같은 데이터를 10 epoch 재사용 |

---

## 5. PPO 핵심 개념 5가지

### 개념 1: Actor-Critic

**파일**: `cartpole_ppo/network.py` — `ActorCritic` 클래스 (line 44)

PPO는 신경망 하나가 두 역할을 동시에 수행한다.

```
                  obs (4차원)
                      │
        ┌─────────────▼─────────────┐
        │  Backbone: Linear → Tanh  │  ← 특징 추출 (공유)
        └──────────┬────────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
   Actor Head           Critic Head
   행동 확률 π(a|s)      상태 가치 V(s)
```

- **Actor**: "어떤 행동을 할까?" → 행동 확률 분포 출력
- **Critic**: "이 상황이 얼마나 좋은가?" → 스칼라 V(s) 출력

Critic 없이는 GAE를 계산할 수 없으므로, 순수 Actor만으로는 PPO를 구성할 수 없다.

---

### 개념 2: On-Policy 데이터 수집

**파일**: `cartpole_ppo/train.py` — rollout 루프 (line 98)

```python
action, log_prob, value = agent.select_action(obs)
next_obs, reward, ...   = env.step(action)
buffer.store(obs, action, log_prob, reward, value, done)
```

**핵심**: `log_prob`(행동 당시의 확률)을 함께 저장한다.
나중에 "그때 정책과 지금 정책이 얼마나 달라졌는가"를 계산하는 재료가 된다.

---

### 개념 3: GAE (Generalized Advantage Estimation)

**파일**: `cartpole_ppo/buffer.py` — `compute_gae()` (line 106)

"이 행동이 얼마나 좋았는가?"를 수치로 만드는 계산.

**TD 오차**:
```
δ_t = r_t + γ·V(s_{t+1})·(1-done) - V(s_t)
    = 실제 받은 것 - Critic이 예측한 것
```

**GAE 어드밴티지** (역방향 누적):
```
A_t = δ_t + γλ·A_{t+1}·(1-done)
```

```python
delta = rewards[t] + gamma * next_v * (1 - dones[t]) - values[t]
gae   = delta + gamma * lam * (1 - dones[t]) * gae
```

| λ 값 | 특성 |
|------|------|
| λ=0 | TD(0): 편향 큼, 분산 작음 |
| λ=1 | Monte Carlo: 편향 작음, 분산 큼 |
| λ=0.95 | 권장값: 균형점 |

- **A_t > 0**: 예상보다 좋은 행동 → 더 많이 해야 함
- **A_t < 0**: 예상보다 나쁜 행동 → 덜 해야 함

---

### 개념 4: PPO Clip Loss (PPO의 핵심)

**파일**: `cartpole_ppo/ppo_agent.py` — `update()` (line 160)

기존 PG는 업데이트 크기에 제한이 없다. PPO는 `clip`으로 제한한다.

```python
ratio = torch.exp(new_log_probs - old_log_probs)
# ratio = 지금 정책 / 그때 정책
# ratio = 1.0 → 변화 없음
# ratio = 1.5 → 확률이 50% 커짐 (큰 변화)

surr1 = ratio * advantages                           # 제한 없는 목표
surr2 = ratio.clamp(1-ε, 1+ε) * advantages          # clip 적용 (ε=0.2)

actor_loss = -torch.min(surr1, surr2).mean()
```

작동 원리 (ε=0.2인 경우):
```
ratio > 1.2 이면 → gradient 차단 → 이 방향으로 더 못 감
ratio < 0.8 이면 → gradient 차단 → 이 방향으로 더 못 감
```

결과: 정책이 한 번에 최대 20%만 변할 수 있다.

---

### 개념 5: Multiple Epochs (같은 데이터 재사용)

**파일**: `cartpole_ppo/ppo_agent.py` — `update()` (line 146)

```python
for _ in range(n_epochs):           # 10번 반복
    for batch in buffer.get_minibatches(batch_size):  # 32개 미니배치
        ...
```

일반 PG: 데이터 수집 → 1번 학습 → 버림
PPO: 데이터 수집 → **10번 반복 학습** → 버림

Clip 덕분에 정책이 너무 많이 바뀌지 않으므로 반복 학습이 안전하다.
학습 효율이 PG 대비 약 10배 향상된다.

---

## 6. 코드 위치 요약

| 개념 | 파일 | 위치 |
|------|------|------|
| Actor-Critic 신경망 | `cartpole_ppo/network.py` | `ActorCritic` 클래스 |
| On-Policy 수집 | `cartpole_ppo/train.py` | rollout 루프 |
| GAE 계산 | `cartpole_ppo/buffer.py` | `compute_gae()` |
| PPO Clip Loss | `cartpole_ppo/ppo_agent.py` | `update()` — ratio, clamp |
| Multiple Epochs | `cartpole_ppo/ppo_agent.py` | `update()` — for n_epochs |
| REINFORCE (비교용) | `cartpole_ppo/pg_agent.py` | `REINFORCEAgent` |

---

## 7. 드론 확장 포인트

| 변경 사항 | 수정 파일 | 구체적 위치 |
|----------|----------|-----------|
| 커스텀 환경 연결 | `train.py` | `CONFIG["env_id"]` |
| 관측/행동 차원 | `train.py` | `CONFIG["obs_dim"]`, `CONFIG["action_dim"]` |
| 연속 행동 공간 | `network.py` | `Categorical` → `Normal` 분포 교체 |
| 하이퍼파라미터 튜닝 | `train.py` | `CONFIG` 딕셔너리 |

PPO 로직(buffer, ppo_agent, train 루프)은 환경이 바뀌어도 수정 불필요.

---

## 참고 자료

- PPO 원논문: Schulman et al., 2017 — "Proximal Policy Optimization Algorithms"
- GAE 원논문: Schulman et al., 2015 — "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
