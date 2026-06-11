import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import scale

def main():
    # 1. 데이터 준비 (비선형 분류 문제인 반달 모양 데이터)
    # 신경망의 성능을 제대로 확인하기 위해 선형으로 나눌 수 없는 데이터를 씁니다.
    print("--- 데이터 준비 중 ---")
    X_full, Y_full = make_moons(noise=0.2, random_state=42, n_samples=300)
    
    # [실무 꿀팁] 딥러닝/변분추론에서는 데이터 스케일링이 생명입니다.
    X = scale(X_full)
    Y = Y_full

    print(f"데이터 개수: {len(Y)}개, 특성 개수: {X.shape[1]}개\n")

    # 2. 베이지안 신경망(BNN) 모델 정의
    # 구조: 입력층(2개) -> 은닉층(5개 노드) -> 출력층(1개 노드)
    n_hidden = 5

    print("베이지안 신경망 모델을 빌드하고 ADVI 엔진으로 학습을 시작합니다...")
    with pm.Model() as bnn_model:
        # [핵심 1] Laplace 사전분포 (가중치를 0으로 밀어붙여 Sparsity 확보)
        # 2차원 입력 -> 5차원 은닉층으로 가는 가중치 행렬
        w_in_1 = pm.Laplace('w_in_1', mu=0, b=1, shape=(X.shape[1], n_hidden))
        # 5차원 은닉층 -> 1차원 출력층으로 가는 가중치
        w_1_out = pm.Laplace('w_1_out', mu=0, b=1, shape=(n_hidden,))

        # 신경망 비선형 연산 (입력층 -> 은닉층 -> 활성화 함수 tanh)
        act_1 = pm.math.tanh(pm.math.dot(X, w_in_1))
        # 신경망 비선형 연산 (은닉층 -> 출력층 -> 활성화 함수 sigmoid)
        act_out = pm.math.sigmoid(pm.math.dot(act_1, w_1_out))

        # 가능도 (0 또는 1 분류이므로 베르누이 분포)
        y_obs = pm.Bernoulli('y_obs', p=act_out, observed=Y)

        # [핵심 2] MCMC 대신 ADVI (변분추론) 알고리즘으로 학습
        # pm.sample() 대신 pm.fit()을 사용합니다.
        approx = pm.fit(n=30000, method='advi')
        print("💡 ADVI 학습 완료!")

        # 학습된 '근사 분포(approx)'에서 최종 사후분포 샘플을 1000개 뽑아냅니다.
        # 이것이 딥러닝 가중치들의 앙상블(Ensemble) 예측을 가능하게 합니다.
        trace = approx.sample(draws=1000)

    # 3. 결과 요약 출력
    print("\n[출력층 가중치 사후분포 요약 통계량]")
    # 은닉층 가중치는 너무 많으므로, 출력층(w_1_out)의 5개 가중치만 확인해 봅니다.
    print(az.summary(trace, var_names=['w_1_out'], kind="stats"))

    # 4. ADVI 손실 함수(ELBO; Evidence Lower Bound) 감소 그래프 시각화
    # 손실값이 우하향하며 바닥을 다졌다면 학습이 성공적으로 된 것입니다.
    plt.plot(approx.hist)
    plt.title('ADVI Optimization: Loss (Negative ELBO)')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()