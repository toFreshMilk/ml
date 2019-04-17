# -*- coding: utf-8 -*-

import numpy as np

X = np.array([6, 8, 10, 14, 18]).reshape(-1,1)
y = np.array([7, 9, 13, 17.5, 18])

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)

# 회귀모델의 평가
# 사이킷 런의 모든 회귀모델 클래스들은 RegressorMixin 클래스
# 를 상속받음
# RegressorMixin 클래스는 결정계수 R2 점수를 계산하는 
# score 메소드를 제공
# 결정계수 R2는 -1 ~ 1 사이의 값을 가지며, 1에 가까울수록
# 좋은 모델임을 확인할 수 있음
print("모델의 평가 : ", model.score(X, y))

# R2(결정계수)의 계산 공식
# 1 - (실제 정답과 예측 값 차이의 제곱값 합계) / 
# (실제 정답과 정답의 평균 값 차이의 제곱합 합계)
pred = model.predict(X)
r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
print("모델의 평가(R2 결정계수 계산) : ", r2)

from sklearn.metrics import r2_score
print("모델의 평가(r2_score 함수) : ", r2_score(y, pred))


















