# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# KNN 알고리즘을 적용하지 않고 예측하는 예제

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


# 테스트를 위한 X 데이터 생성
X_test = np.array([[155, 70]])

# 테스트 데이터와 학습 데이터 사이의 유클리드 거리 계산
# (T1, T2)와 (R1, R2) 사이의 유클리드 거리 계산법
# A = ((T1 - R1)의 제곱 + (T2 - R2)의 제곱) 
# 유클리드 거리 = A의 제곱근

# np.sum : 매개변수의 총 합계를 구하는 함수
# axis 매개변수의 값을 지정하지 않으면 전체 함계가 반환
# axis = 0 인경우 열의 함계를 반환
# axis = 1 인경우 행의 함계를 반환

# np.sqrt : 매개변수의 제곱근 값을 반환
distances = np.sqrt(np.sum((X_train - X_test) ** 2, axis=1))
print(f"테스트 데이터와 학습 데이터 사이의 거리 : \n{distances}")

# 테스트 데이터와 가장 가까운 학습 데이터 3개의 인덱스 추출
# numpy 배열의 argsort 메소드
# 배열 내부를 오름차순으로 정렬했을때의 인덱스 값을 반환
# 내림차순의 경우 argsort()[::-1][:3]
nearest_indices = distances.argsort()[:3]
print(f"nearest_indices : {nearest_indices}")

# 테스트 데이터와 가장 가까운 학습 데이터 3개의 인덱스를 활용하여
# 각 학습 데이터의 정답 데이터를 추출
# np.take 함수는 1번째 매개변수로 전달된 배열로부터
# 2번째 매개변수로 전달된 인덱스에 해당되는 요소를 반환
nearest_genders = np.take(y_train, nearest_indices)
print(f"nearest_genders : {nearest_genders}")

# 값의 개수를 확인할 수 있는 Counter 클래스
from collections import Counter
counter = Counter(np.take(y_train, distances.argsort()[:3]))
# 테스트 데이터와 가장 인접한 이웃 데이터에서 
# 가장 빈도수가 높은 값을 추출
print(counter)
print(counter.most_common())
print(counter.most_common()[0][0])






























