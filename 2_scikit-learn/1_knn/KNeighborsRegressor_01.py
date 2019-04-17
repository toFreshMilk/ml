# -*- coding: utf-8 -*-

import numpy as np

# KNN 알고리즘을 활용한 회기 분석 예제

# 학습 데이터 X
# 키와 성별 정보
X_train = np.array([
    [158, 1],
    [170, 1],
    [183, 1],
    [191, 1],
    [155, 0],
    [163, 0],
    [180, 0],
    [158, 0],
    [170, 0]
])
# 학습 데이터 y
# 몸무게 정보
y_train = [64, 86, 84, 80, 49, 59, 67, 54, 67]

# 테스트 데이터 X
X_test = np.array([
    [160, 1],
    [196, 1],
    [168, 0],
    [177, 0]
])
# 테스트 데이터 y
y_test = [66, 87, 68, 74]

# KNN 알고리즘을 활용하여 수치 값을 예측할 수 있는 
# KNeighborsRegressor 클래스
from sklearn.neighbors import KNeighborsRegressor

# 탐색할 인접 데이터 개수
K = 3
# KNN 알고리즘을 활용하여 수치 예측을 할 수 있는
# KNeighborsRegressor 객체 생성
model = KNeighborsRegressor(n_neighbors=K)
# fit 메소드를 호출하여 학습데이터 지정
# (데이터 학습은 진행하지 않음)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(f'예측된 몸무게 : {predictions}')
print(f'실제 몸무게 : {y_test}')

# KNeighborsRegressor 클래스의 
# score 메소드를 통한 모델 평가
# 회귀 분석 모델의 경우 R2(결정계수) 값을 반환

# R2(결정계수) 계산 공식
# 1 - (실제 정답과 예측 값 차이의 제곱 값 합계) / 
# (실제 정답과 정답의 평균 값 차이의 제곱 값 합계)

# 분석결과의 이해
# 1에 가까울수록 좋은 예측을 보이는 모델
# 0에 가까울수록 정답의 평균 수치를 예측(최적화가 되지 않음을 알림)
# -값이 나오는 경우 정답의 평균 수치에서 멀어짐을 나타냄
# (예측 결과를 사용할 수 없음)
print(f'모델 평가(TRAIN) : {model.score(X_train, y_train)}')
print(f'모델 평가(TEST) : {model.score(X_test, y_test)}')

# r2_score 함수를 사용한 모델 평가
from sklearn.metrics import r2_score
print(f'R2 점수(TEST) : {r2_score(y_test, predictions)}')





















