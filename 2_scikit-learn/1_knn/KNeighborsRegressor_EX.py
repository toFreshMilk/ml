# -*- coding: utf-8 -*-

# 사이킷 런의 load_boston 데이터를
# KNN 알고리즘을 사용하여 분석한 후, 
# 모델의 평가점수를 출력하세요

import pandas as pd
from sklearn.datasets import load_boston

boston = load_boston()

X_df = pd.DataFrame(boston.data, columns=boston.feature_names)
#X_df.columns = boston.feature_names
y_df = pd.Series(boston.target)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
train_test_split(X_df.values, y_df.values, random_state=1)

from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(n_neighbors=4).fit(X_train, y_train)
#model.fit(X_train, y_train)

print("학습 데이터 평가 : ", model.score(X_train, y_train))
print("테스트 데이터 평가 : ", model.score(X_test, y_test))

print("=" * 20)
# 회귀모델의 평가에 사용되는 지표
# 평균제곱오차, 평균절대오차
# 평균제곱오차 : (실제정답 - 예측값)의 제곱의 합계를 구한 후, 평균
# 평균절패오차 : (실제정답 - 예측값)의 절대값의 합계를 구한 후, 평균
from sklearn.metrics import mean_squared_error, mean_absolute_error

pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

# 평균제곱오차의 값을 반환하는 mean_squared_error 함수
print("학습데이터에 대한 평균제곱오차 : ", 
      mean_squared_error(y_train, pred_train))
print("테스트데이터에 대한 평균제곱오차 : ", 
      mean_squared_error(y_test, pred_test))

# 평균절대오차의 값을 반환하는 mean_absolute_error 함수
print("학습데이터에 대한 평균절대오차 : ", 
      mean_absolute_error(y_train, pred_train))
print("테스트데이터에 대한 평균절대오차 : ", 
      mean_absolute_error(y_test, pred_test))














