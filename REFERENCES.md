# 참고문헌

---

## 논문

### 핵심 논문

| 약칭 | 논문 | 저자 | 연도 | 이 프로젝트에서의 역할 |
|------|------|------|------|----------------------|
| **PPO** | [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) | Schulman et al. | 2017 | PPO Clip Loss, Multiple Epochs |
| **GAE** | [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) | Schulman et al. | 2016 | `buffer.py`의 GAE 구현 |
| **REINFORCE** | Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning | Williams | 1992 | `pg_agent.py`의 Policy Gradient 기반 |

### 관련 논문

| 논문 | 저자 | 연도 | 내용 |
|------|------|------|------|
| [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477) | Schulman et al. | 2015 | PPO의 전신. KL constraint → PPO는 이를 Clip으로 단순화 |
| [Actor-Critic Algorithms](https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf) | Konda & Tsitsiklis | 2000 | Actor-Critic 구조의 이론적 기반 |
| [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf) | Sutton et al. | 2000 | Policy Gradient Theorem |

---

## 서적

| 서적 | 저자 | 관련 내용 |
|------|------|----------|
| [Reinforcement Learning: An Introduction (2nd ed.)](http://incompleteideas.net/book/the-book.html) | Sutton & Barto | RL 기본 개념, Policy Gradient, Actor-Critic |
| [Deep Reinforcement Learning Hands-On (2nd ed.)](https://www.packtpub.com/product/deep-reinforcement-learning-hands-on-second-edition/9781838826994) | Lapan | PPO 구현 실습 참고 |

---

## 라이브러리 & 문서

| 라이브러리 | 문서 | 이 프로젝트에서의 역할 |
|-----------|------|----------------------|
| **Gymnasium** | [gymnasium.farama.org](https://gymnasium.farama.org/) | RL 환경 표준 인터페이스, CartPole-v1 내장 환경, 커스텀 환경 구현 가이드 |
| **PyTorch** | [pytorch.org/docs](https://pytorch.org/docs/stable/) | Actor-Critic 신경망, 자동 미분, 옵티마이저 |
| **Pygame** | [pygame.org/docs](https://www.pygame.org/docs/) | 드론 투하 환경 실시간 렌더링 |
| **TensorBoard** | [tensorflow.org/tensorboard](https://www.tensorflow.org/tensorboard) | 학습 지표 실시간 시각화 |
| **Matplotlib** | [matplotlib.org](https://matplotlib.org/) | 학습 곡선, 비교 분석 차트 |
| **NumPy** | [numpy.org/doc](https://numpy.org/doc/) | 수치 계산, 버퍼 관리 |

### Gymnasium 커스텀 환경 관련

| 자료 | 링크 |
|------|------|
| Custom Environment 공식 가이드 | [gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/) |
| Spaces 문서 | [gymnasium.farama.org/api/spaces](https://gymnasium.farama.org/api/spaces/) |
| CartPole-v1 소스코드 | [github.com/Farama-Foundation/Gymnasium/.../cart_pole.py](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cart_pole.py) |

---

## 코드 참고

| 자료 | 링크 | 참고 내용 |
|------|------|----------|
| CleanRL (PPO 구현) | [github.com/vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl) | 단일 파일 PPO 레퍼런스 구현 |
| Spinning Up (OpenAI) | [spinningup.openai.com](https://spinningup.openai.com/en/latest/) | PPO/VPG 알고리즘 설명 + 구현 |
| The 37 Implementation Details of PPO | [iclr-blog-track.github.io/2022/03/25/ppo-implementation-details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) | PPO 구현 세부사항 37가지 정리 |

---

## 이 프로젝트의 코드-논문 매핑

| 코드 파일 | 핵심 구현 | 근거 논문/자료 |
|----------|----------|--------------|
| `network.py` | Orthogonal 초기화, 공유 Backbone | CleanRL, PPO impl. details |
| `buffer.py` — `compute_gae()` | GAE 역방향 누적 | GAE 논문 (Schulman 2016) |
| `ppo_agent.py` — Clip Loss | `ratio.clamp(1-ε, 1+ε)` | PPO 논문 (Schulman 2017) |
| `ppo_agent.py` — Entropy Bonus | `-entropy_coef × H[π]` | PPO 논문 |
| `pg_agent.py` — REINFORCE | `-(log_prob × G_t).sum()` | Williams 1992 |
| `env.py` — Gymnasium 인터페이스 | `reset()`, `step()`, `render()` | Gymnasium 공식 가이드 |
| `env.py` — 해석적 착지 계산 | `landing_x = drop_x + v × t` | 기초 물리학 (포물선 운동) |
