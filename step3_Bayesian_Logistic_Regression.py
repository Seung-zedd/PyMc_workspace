import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from sklearn import datasets

def main():
    # 1. 데이터 준비: scikit-learn의 Iris(붓꽃) 데이터셋 불러오기
    print("--- 데이터 준비 중 ---")
    iris = datasets.load_iris()
    X_full = iris.data
    y_full = iris.target

    # 이진 분류를 위해 0번(Setosa)과 1번(Versicolor) 클래스만 추출
    # 특성(Feature)도 2개(꽃받침 길이, 꽃받침 너비)만 사용하여 직관성을 높입니다.
    mask = y_full < 2
    X = X_full[mask, :2]
    Y = y_full[mask]

    # [실무 꿀팁] 데이터 표준화 (NUTS 샘플링 속도와 안정성을 극대화하기 위해 필수!)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_scaled = (X - X_mean) / X_std

    print(f"데이터 개수: {len(Y)}개, 특성 개수: {X_scaled.shape[1]}개\n")

    # 2. PyMC 로지스틱 회귀 모델 정의
    print("베이지안 로지스틱 회귀 모델을 정의하고 샘플링을 시작합니다...")
    with pm.Model() as logistic_model:
        # 사전분포 (Prior): 특성이 2개이므로 가중치(w)도 2개가 필요합니다.
        # shape=2를 주어 w[0], w[1] 두 개의 변수를 한 번에 선언합니다.
        w = pm.Normal('w', mu=0, sigma=10, shape=2)
        b = pm.Normal('b', mu=0, sigma=10)
        
        # 선형 결합 (z = w0*x0 + w1*x1 + b)
        # pm.math.dot을 사용하여 행렬 내적을 깔끔하게 처리합니다.
        z = pm.math.dot(X_scaled, w) + b
        
        # 시그모이드 함수 적용 (결과값 z를 0~1 사이의 확률 p로 변환)
        p = pm.Deterministic('p', pm.math.sigmoid(z))
        
        # 가능도 (Likelihood): 0 또는 1의 이진 분류이므로 베르누이 분포 사용
        y_obs = pm.Bernoulli('y_obs', p=p, observed=Y)
        
        # 3. MCMC 샘플링 (NUTS 자동 가동)
        trace = pm.sample(draws=2000, tune=1000, return_inferencedata=True)

    # 4. 결과 요약 출력
    print("\n[사후분포 요약 통계량]")
    print(az.summary(trace, var_names=['w', 'b'], kind="stats"))

    # 5. Trace Plot 시각화
    az.plot_trace(trace, var_names=['w', 'b'])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()