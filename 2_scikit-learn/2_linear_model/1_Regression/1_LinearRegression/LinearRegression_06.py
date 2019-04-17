# -*- coding: utf-8 -*-

import numpy as np

X = np.array([6, 8, 10, 14, 18]).reshape(-1,1)
y = np.array([7, 9, 13, 17.5, 18])

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)

# LinearRegression 클래스의 가중치(기울기), 절편 계산방법
# 1. 입력데이터 X의 분산 값을 계산
# - 입력데이터 X의 산포 수치를 계산
# - 모든 입력데이터가 동일하다면 분산은 0
# - 분산 수치가 작다면 데이터들이 평균 부근에 밀집
# - 분산 수치가 크다면 데이터들이 평균에서 멀리 위치하고 있음 

# 계산식 : 각 X 데이터를 X 데이터의 평균으로 감소시킨 후,
# 제곱하여 합계를 구함. 합계에 대해서 데이터 개수 -1로 나눈 값이
# 분산의 값이 됨

X_mean = np.mean(X)
print("X 데이터의 평균 : ", X_mean)
variance = np.sum((X - X_mean) ** 2) / (len(X) - 1)
print("X 데이터의 분산 : ", variance)

# 2. 입력 데이터 X와 정답데이터 y 사이의 공분산 값을 계산
# - 공분산 : 두개의 변수가 함께 변화하는 수치를 측정하기 위한 방법
# - 한 변수가 증가할 때 다른 변수도 증가한다면 공분산은 양수
# - 한 변수가 증가할 때 다른 변수가 감소한다면 공분산은 음수

# 계산식 : 두 변수에 대해서 각각의 데이터의 평균만큼 감소시킨 후,
# 서로 곱한 결과의 합계에 대해서 데이터 크기 - 1 로 나누 값
X_mean = np.mean(X)
y_mean = np.mean(y)

covariance = (X.reshape(-1) - X_mean) * (y - y_mean)
covariance = np.sum(covariance) / (len(y) - 1)
print("공분산 수치 : ", covariance)

# 3. X 데이터의 분산 값과 X, y 데이터의 공분산 값을 계산한 후
# 가중치(기울기)를 계산함
# - 계산식 : 공분산 / 분산
weight = covariance / variance
print("가중치(기울기) : ", weight)
print("가중치(coef_) : ", model.coef_)

# 4. 가중치의 값을 계산한 후, 절편의 값을 계산
# - 정답 데이터(y)의 평균 - (입력 데이터(X)의 평균 * 가중치)
bias = np.mean(y) - (np.mean(X) * weight)
print("절편 : ", bias)
print("절편(intercept_) : ", model.intercept_)















