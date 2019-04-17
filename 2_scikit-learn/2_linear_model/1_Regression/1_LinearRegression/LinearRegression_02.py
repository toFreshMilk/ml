# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# 피자 크기에 따른 가격 데이터
# 사이킷런의 모든 예측기는 입력데이터의 형태를
# 2차원 배열로 가정하고 있기 때문에
# 전달할 입력데이터를 reshape를 통해서
# 형태를 변경
X = np.array([6, 8, 10, 14, 18]).reshape(-1,1)
# 정답 데이터는 일차원 배열로 선언
y = np.array([7, 9, 13, 17.5, 18])

# 입력 데이터에 대한 정답 데이터(수치)를
# 선형모델을 사용하여 예측할 수 있는 
# LinearRegression 모델 클래스를 import
from sklearn.linear_model import LinearRegression

# LinearRegression 클래스의 객체 생성 매개변수
# copy_X : 입력데이터를 복사 여부
# fit_intercept : 절편의 값을 계산 여부
# normalize : 정규화 여부
# n_jobs : 데이터 분석에 사용할 코어 개수
# (기본값 1, -1로 입력하는 경우 사용가능한 모든 코어를 사용)
model = LinearRegression()

# 사이킷런의 모든 예측기들은 fit 메소드를 제공하며
# fit 메소드는 입력된 X, y 데이터를 학습하는 기능을 제공
# LinearRegression 클래스의 fit 메소드는
# 입력된 데이터에 최적화되는 선형방정식을 계산
# 선형 방정식 : x1 * w1 + x2 * w2 ... + b
model.fit(X, y)

print("학습데이터의 평가점수 : ", model.score(X,y))

# 테스트 데이터를 생성하여 모델의 예측값을 확인하세요
# 12인치, 20인치 인 경우의 예측 가격을 출력하세요.

# 사이킷런의 모든 예측기들은 fit 메소드가 실행된 후
# predict 메소드를 통해서 입력된 데이터에 따른
# 예측 결과를 반환할 수 있음.
# predict 메소드는 입력 데이터로 2차원 배열 형태를 가정하며
# 반환되는 값은 입력된 2차원 배열의 행의 수와 일치하는 일차원 배열
X_test = np.array([12, 20]).reshape(-1,1)
pred = model.predict(X_test)

print("12인치에 대한 예측 결과 : ", pred[0])
print("20인치에 대한 예측 결과 : ", pred[1])

print(f"12 인치 피자 가격 예측 : {pred[0]:.2f}")
print(f"20 인치 피자 가격 예측 : {pred[1]:.2f}")

























