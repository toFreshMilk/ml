# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# KNN 알고리즘을 활용하여 예측하는 예제

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
y_train = ['male', 'male', 'male', 'male', 'female', 
           'female', 'female', 'female', 'female']

plt.figure()
plt.title('Human Heights and Weights by Gender')
plt.xlabel('Height in cm')
plt.ylabel('Weight in kg')

for i, x in enumerate(X_train):
    plt.scatter(x[0], x[1], c='k', 
                marker='x' if y_train[i] == 'male' else 'D')
plt.grid(True)
plt.show()

# KNN 알고리즘을 구현하고 있는 KNeighborsClassifier 클래스
from sklearn.neighbors import KNeighborsClassifier

# 탐색할 최근접 이웃의 개수를 지정
K = 3

# KNeighborsClassifier 객체를 생성 시
# 최근접 이웃의 개수를 생성자로 전달
# (기본값은 5)
model = KNeighborsClassifier(n_neighbors=K)

# fit 메소드를 사용하여 데이터를 전달
# 학습을 진행하지 않음(게으른 학습 방법)
model.fit(X_train, y_train)

# predict 메소드를 통해 예측 결과를 반환받음
predicted = model.predict(np.array([155, 70]).reshape(1, -1))[0]

print(predicted)

plt.figure()
plt.title('Human Heights and Weights by Gender')
plt.xlabel('Height in cm')
plt.ylabel('Weight in kg')

for i, x in enumerate(X_train):
    plt.scatter(x[0], x[1], c='k', 
                marker='x' if y_train[i] == 'male' else 'D')
    
plt.scatter([155], [70], c='r', marker='v')
plt.grid(True)
plt.show()

# 학습 결과를 테스트하기 위한 데이터 셋 설정
X_test = np.array([
    [168, 65],
    [180, 96],
    [160, 52],
    [169, 67]
])
    
# 테스트 데이터 셋의 정답(라벨)
y_test = ['male', 'male', 'female', 'female']

predictions = model.predict(X_test)
print(f'예측 결과 : {predictions}')
print(f'예측 정확도 : {model.score(X_test, y_test)}')










