# 📊 PyMC & Bayesian Data Science Lab

이 저장소는 PyMC 라이브러리를 활용한 **베이지안 통계 분석 및 머신러닝 모델링**의 학습 과정을 기록하고 실무 응용 능력을 배양하기 위한 workspace입니다.

단순히 라이브러리 사용법을 익히는 것에 그치지 않고, 모델의 수학적 원리(선형 결합, 시그모이드, 확률 분포 등)와 실무적 시사점을 마크다운 문서로 체계화하여 관리합니다.

## 🚀 주요 학습 내용

1.  **MCMC 알고리즘 이해**: `Metropolis` 알고리즘의 한계와 고성능 `NUTS` 샘플러의 작동 원리 분석.
2.  **베이지안 선형 회귀**: 연속형 수치 예측 모델 구축 및 사후분포(Posterior)를 통한 불확실성 추정.
3.  **베이지안 로지스틱 회귀**: 시그모이드 함수와 베르누이 분포를 결합한 이진 분류 모델 구현 및 결정 경계선 시각화.
4.  **데이터 전처리 최적화**: NUTS 수렴도 향상을 위한 데이터 표준화(Scaling) 기여도 확인.

## 🎯 향후 주요 학습 로드맵 (Upcoming)

### 5. **베이지안 신경망 (BNN)** (난이도: 상 ★★★)

- **목표**: 심층신경망(딥러닝)에 사전분포/사후분포 개념 도입하기
- **주요 내용**:
  - 가중치 행렬의 희소성(Sparsity)을 위해 **Laplace 사전분포**를 부여한 신경망 구축.
  - MCMC의 속도 한계를 극복하기 위해 자동미분 기반 **변분추론(pm.ADVI)** 학습 및 적용.
  - 여러 사후분포 샘플들의 평균(Ensemble)을 통한 최종 예측 정확도 및 불확실성 산출.

## 📂 프로젝트 구조

- `docs/`: 각 분석 단계별 핵심 원리 및 시사점이 정리된 TIL(Today I Learned) 리포트.
- `assets/visual_data/`: 모델 구조 그래프, Trace Plot, 데이터 분포도 등 시각화 산출물.
- `step*.py`: 단계별 분석 소스 코드.

## 🛠️ 주요 스택 및 알고리즘

- **Language**: Python
- **Library**: `PyMC`, `ArviZ`, `Matplotlib`, `scikit-learn`
- **Algorithm**: MCMC (Metropolis, NUTS)
- **Model**: Linear Regression, Logistic Regression
