import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

def main():
    # 1. 가상 데이터 생성 (y = 2x + 1 + 에러)
    np.random.seed(42)
    X = np.linspace(0, 10, 100)
    true_w = 2.0 # 기울기
    true_b = 1.0 # 절편
    # 실제 데이터에 노이즈(표준편차 1.5)를 추가합니다.
    y_noise = np.random.normal(0, 1.5, size=100)
    Y = true_w * X + true_b + y_noise

    print("--- 데이터 준비 완료 ---")
    print(f"실제 정답 -> 기울기(w): {true_w}, 절편(b): {true_b}\n")

    # 2. PyMC 베이지안 선형 회귀 모델 정의
    with pm.Model() as model:
        # 사전분포(Prior): 기울기와 절편이 어떤 값일지 정규분포로 넓게 가정합니다.
        w = pm.Normal('w', mu=0, sigma=10)
        b = pm.Normal('b', mu=0, sigma=10)
        
        # 데이터의 노이즈(표준편차) 크기에 대한 사전분포
        sigma = pm.HalfNormal('sigma', sigma=5)
        
        # 선형 회귀 공식 (Deterministic 변수는 계산된 결론값을 의미합니다)
        mu = pm.Deterministic('mu', w * X + b)
        
        # 가능도(Likelihood): 실제 관측치 Y가 선형 공식(mu)을 중심으로 정규분포를 따른다고 설정
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=Y)
        
        # [시각화 포인트] 모델 구조를 그래프 이미지로 저장 및 출력
        # 코드가 있는 폴더에 'model_structure.png' 파일이 생성됩니다.
        graph = pm.model_to_graphviz(model)
        graph.render("model_structure", format="png", cleanup=True)
        print("💡 모델 구조 그래프('model_structure.png')가 저장되었습니다.")

        # 3. MCMC 샘플링 (MH 알고리즘 대신 step 인자를 제거함으로써 자동으로 NUTS 사용)
        print("샘플링을 시작합니다...")
        trace = pm.sample(draws=2000, tune=1000, return_inferencedata=True)

    # 4. 결과 요약 출력
    print("\n[사후분포 요약 통계량]")
    # mu는 모든 데이터 포인트의 계산값이므로 제외하고 파라미터(w, b, sigma)만 봅니다.
    print(az.summary(trace, var_names=['w', 'b', 'sigma'], kind="stats"))

    # 5. Trace Plot 시각화
    az.plot_trace(trace, var_names=['w', 'b', 'sigma'])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()