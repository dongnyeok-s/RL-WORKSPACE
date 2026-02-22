# RL Workspace — 강화학습 실습 프로젝트

REINFORCE(Vanilla Policy Gradient)와 PPO(Proximal Policy Optimization)를 직접 구현하고,
두 알고리즘을 비교 분석하는 강화학습 학습용 프로젝트.

---

## 프로젝트 구성

| 디렉토리 | 환경 | 핵심 내용 |
|----------|------|----------|
| [`cartpole/`](cartpole/) | CartPole-v1 (Gymnasium 내장) | PPO/REINFORCE 구현 + 비교 실험 |
| [`drone_drop/`](drone_drop/) | 2D 드론 투하 (Gymnasium 커스텀) | 커스텀 환경 설계 + PPO vs REINFORCE 성능 비교 |

### 학습 순서

```
1. cartpole/  — PPO 핵심 개념 학습 (Actor-Critic, GAE, Clip Loss)
       ↓
2. drone_drop/ — 커스텀 환경 설계 + 실전 적용
```

---

## 전체 파일 구조

```
RL Workspace/
├── README.md                          ← 이 파일
├── requirements.txt                   ← 통합 의존성
│
├── cartpole/                          ── 실험 1: CartPole ──
│   ├── README.md                      프로젝트 설명
│   ├── docs/
│   │   ├── PPO_개념정리.md            PPO 이론 정리
│   │   └── REINFORCE_vs_PPO_비교분석.md  실험 결과 비교 문서
│   ├── cartpole_ppo/
│   │   ├── network.py                 Actor-Critic 신경망
│   │   ├── buffer.py                  Rollout Buffer + GAE
│   │   ├── ppo_agent.py               PPO Clip Loss 업데이트
│   │   ├── pg_agent.py                REINFORCE 에이전트
│   │   ├── train.py                   PPO 학습 스크립트
│   │   ├── train_pg.py                REINFORCE 학습 스크립트
│   │   └── evaluate.py                모델 평가 + 렌더링
│   └── results/                       학습 결과 (모델, 그래프, 로그)
│
└── drone_drop/                        ── 실험 2: 드론 투하 ──
    ├── README.md                      프로젝트 설명
    ├── make_analysis.py               PPO vs REINFORCE 비교 차트 생성
    ├── drone_drop_ppo/
    │   ├── env.py                     ★ 커스텀 Gymnasium 환경 + Pygame 렌더링
    │   ├── network.py                 Actor-Critic 신경망
    │   ├── buffer.py                  Rollout Buffer + GAE
    │   ├── ppo_agent.py               PPO Clip Loss 업데이트
    │   ├── pg_agent.py                REINFORCE 에이전트
    │   ├── train.py                   PPO 학습 스크립트
    │   ├── train_pg.py                REINFORCE 학습 스크립트
    │   └── evaluate.py                모델 평가 + Pygame 시각화
    └── results/                       학습 결과 (모델, 그래프, 로그)
```

---

## 빠른 시작

### 1. 환경 설정

```bash
pip install -r requirements.txt
```

### 2. CartPole 실험

```bash
# PPO 학습
cd cartpole
python -m cartpole_ppo.train

# REINFORCE 학습 (PPO와 비교)
python -m cartpole_ppo.train_pg

# 학습된 모델 평가 (Pygame 렌더링)
python -m cartpole_ppo.evaluate
python -m cartpole_ppo.evaluate --compare     # 미학습 vs 학습 비교
```

### 3. 드론 투하 실험

```bash
# PPO 학습
cd drone_drop
python -m drone_drop_ppo.train

# REINFORCE 학습
python -m drone_drop_ppo.train_pg

# Pygame으로 시각화
python -m drone_drop_ppo.evaluate                  # PPO 단독
python -m drone_drop_ppo.evaluate --compare-pg     # REINFORCE vs PPO 비교

# 비교 분석 차트 생성
python make_analysis.py
```

### 4. TensorBoard 실시간 모니터링

```bash
# CartPole
tensorboard --logdir cartpole/results/tensorboard

# 드론 투하
tensorboard --logdir drone_drop/results/tensorboard
```

---

## 알고리즘 비교 요약

### CartPole-v1 (500K steps)

| | REINFORCE | PPO |
|---|---|---|
| 최종 평균 보상 | 500.0 (만점) | 500.0 (만점) |
| 수렴 속도 | ~300K steps | ~80K steps |
| 학습 안정성 | 불안정 (큰 변동) | 안정적 |

### 드론 투하 (300K steps)

| | REINFORCE | PPO |
|---|---|---|
| 평균 보상 | 6.30 | 13.67 |
| 평균 착지 거리 | 139.6px | 15.8px |
| 성공률 (< 30px) | 0% | 100% |

> CartPole에서는 두 알고리즘 모두 만점에 도달하지만,
> 드론 투하처럼 복잡한 환경에서는 PPO의 이점이 극명하게 나타난다.

---

## 핵심 구현 포인트

### PPO 5대 구성요소

| 구성요소 | 파일 | 핵심 코드 |
|---------|------|----------|
| Actor-Critic 신경망 | `network.py` | `ActorCritic.forward()` |
| Rollout Buffer + GAE | `buffer.py` | `compute_gae()` |
| Clip Loss | `ppo_agent.py` | `ratio.clamp(1-ε, 1+ε)` |
| Multiple Epochs | `ppo_agent.py` | `for _ in range(n_epochs)` |
| On-Policy 수집 | `train.py` | rollout 루프 |

### 드론 커스텀 환경

| 설계 포인트 | 내용 |
|------------|------|
| 관측 공간 | 4차원 정규화 벡터 (위치, 속도, 상대거리, **타이밍 신호**) |
| 보상 설계 | 투하 즉시 해석적 착지 계산 → 보상 지연 문제 해결 |
| Pygame 렌더링 | 드론, 목표, 패키지 궤적, 최적 투하선 시각화 |

---

## 기술 스택

- **Python 3.11+**
- **PyTorch** — 신경망 학습
- **Gymnasium** — RL 환경 표준 인터페이스
- **Pygame** — 실시간 시각화
- **TensorBoard** — 학습 지표 모니터링
- **Matplotlib** — 결과 차트 생성

---

## 문서

| 문서 | 내용 |
|------|------|
| [REFERENCES.md](REFERENCES.md) | 참고 논문, 서적, 라이브러리, 코드-논문 매핑 |
| [cartpole/docs/PPO_개념정리.md](cartpole/docs/PPO_개념정리.md) | RL 기초 ~ PPO 이론 정리 |
| [cartpole/docs/REINFORCE_vs_PPO_비교분석.md](cartpole/docs/REINFORCE_vs_PPO_비교분석.md) | CartPole 실험 결과 비교 |
| [drone_drop/README.md](drone_drop/README.md) | 드론 투하 환경 설계 + 실험 결과 |
