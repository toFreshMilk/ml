# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# KNN 알고리즘 (K-Nearest Neighbor)
# K-최근접 이웃
# 데이터 간 거리를 측정하여 가장 가까운 K개의 이웃의 
# 결과 데이터 중 빈도수가 높은 결과로 추론하는 알고리즘
# 단순하지만 여러 응용분야에서 활용되고 있으며,
# 검색 및 추천 시스템에서도 활용됨
# 검색할 이웃의 개수(k)의 값을 매개변수로 지정하며
# 동률인 경우를 제거하기 위해서 일반적으로 홀수로 지정함

# 게으른 학습 방법을 지향하는 KNN
# 일반적인 머신러닝 알고리즘과는 달리 
# 사전에 데이터를 학습하지 않고, 저장만 함
# 실제 예측을 하는 시점(predict 메소드 실행)에
# 각 데이터들을 순회하며 예측을 생성함

# 신장과 몸무게의 데이터를 저장하고 있는 X 데이터
X_train = np.array([
    [158, 64],
    [170, 86],
    [183, 84],
    [191, 80],
    [155, 49],
    [163, 59],
    [180, 67],
    [158, 54],
    [170, 67]
])
# 신장과 몸무게 정보를 이용하여 예측해야하는 성별 정보
y_train = ['male', 'male', 'male', 'male', 'female', 
           'female', 'female', 'female', 'female']

plt.figure()
plt.title('Human Heights and Weights by Gender')
plt.xlabel('Height in cm')
plt.ylabel('Weight in kg')

for i, x in enumerate(X_train):
    plt.scatter(x[0], x[1], c='k', marker='x' if y_train[i] == 'male' else 'D')
plt.grid(True)
plt.show()














