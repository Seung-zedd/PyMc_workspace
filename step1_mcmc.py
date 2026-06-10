import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

def main():
    # 1. 가상 데이터 생성 (진짜 평균 = 5.0, 표준편차 = 1.0)
    np.random.seed(42)
    true_mu = 5.0
    true_sigma = 1.0
    data = np.random.normal(loc=true_mu, scale=true_sigma, size=100)

    print(f"--- 데이터 준비 완료 ---")
    print(f"생성된 데이터의 표본 평균: {np.mean(data):.4f}\n")

    # 2. PyMC 모델 정의
    print("베이지안 모델을 정의하고 샘플링을 시작합니다...")
    with pm.Model() as model:
        # 사전분포 (Prior): 평균 mu가 어디쯤 있을지 모른다는 가정하에 Cauchy 분포 설정
        mu = pm.Cauchy('mu', alpha=0, beta=5)
        
        # 가능도 (Likelihood): 데이터가 정규분포를 따른다고 가정
        y = pm.Normal('y', mu=mu, sigma=true_sigma, observed=data)
        
        # 3. Metropolis-Hastings 알고리즘으로 샘플링 수행
        step = pm.Metropolis()
        # draws: 기록할 샘플 수, tune: 초반에 버릴 적응용 샘플 수
        trace = pm.sample(draws=2000, tune=1000, step=step, return_inferencedata=True)

    print("\n--- 샘플링 완료! 결과를 시각화합니다. ---")

    # 4. 사후분포 요약 통계량 출력
    summary = az.summary(trace, kind="stats")
    print("\n[사후분포 요약 통계량]")
    print(summary)
    
    # 5. Trace Plot 시각화
    az.plot_trace(trace)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()