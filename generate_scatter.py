import matplotlib.pyplot as plt
from sklearn import datasets
import os

def main():
    # 1. 데이터 준비
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # 이진 분류를 위해 0번(Setosa)과 1번(Versicolor) 클래스만 추출
    # 특성(Feature)은 2개(꽃받침 길이, 꽃받침 너비)만 사용
    mask = y < 2
    X_filtered = X[mask, :2]
    y_filtered = y[mask]

    # 2. 스캐터 플롯(산점도) 그리기
    plt.figure(figsize=(8, 6))
    
    # Setosa (0) 시각화
    plt.scatter(X_filtered[y_filtered == 0, 0], X_filtered[y_filtered == 0, 1], 
                color='blue', label='Setosa (Class 0)', alpha=0.7, edgecolors='black', s=80)
    
    # Versicolor (1) 시각화
    plt.scatter(X_filtered[y_filtered == 1, 0], X_filtered[y_filtered == 1, 1], 
                color='orange', label='Versicolor (Class 1)', alpha=0.7, edgecolors='black', s=80)

    # 그래프 꾸미기
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.title('Iris Dataset: Setosa vs Versicolor (Sepal Features)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # 3. 이미지 저장
    os.makedirs('assets/visual_data', exist_ok=True)
    plt.savefig('assets/visual_data/iris_scatter.png', dpi=300)
    print("스캐터 플롯 이미지가 성공적으로 저장되었습니다: assets/visual_data/iris_scatter.png")

if __name__ == "__main__":
    main()