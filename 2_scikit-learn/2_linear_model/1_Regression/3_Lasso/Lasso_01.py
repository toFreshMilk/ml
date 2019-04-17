# -*- coding: utf-8 -*-

import pandas as pd

fname = '../../data/score.csv'
df = pd.read_csv(fname)

df.drop('name', axis=1, inplace=True)

X_df = df.iloc[:, 1:]
y_df = df.iloc[:, 0]

X_train = X_df.values
y_train = y_df.values

# 선형 모델에 L1 제약 조건을 추가한 Lasso 클래스
# L1 제약 조건 : 모든 특성 데이터 중 특정 특성에 대해서만 가중치의 값을
# 할당하는 제약조건
# (대다수의 특성은 0으로 제약)
# L1 제약 조건은 특성 데이터가 많은 데이터를 학습하는 경우 
# 빠르게 학습을 할 수 있는 장점을 가짐
# 모든 특성 데이터 중 중요도가 높은 특성을 구분할 수 있음
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# 분석 모델 클래스의 객체 생성
lr_model = LinearRegression().fit(X_train, y_train)
ridge_model = Ridge(alpha=100).fit(X_train, y_train)
# Lasso 클래스의 하이퍼 파라메터 alpha
# alpha의 값이 커질수록 제약을 크게 설정
# (alpha의 값이 커질수돌 대다수의 특성에 대한 가중치의 값이 0으로 수렴)
# alpha의 값이 작아질수록 제약이 약해짐
# (alpha의 값이 작아질수록 적은 수의 특성에 대한 가중치의 값은 0으로 수혐)
lasso_model = Lasso(alpha=5).fit(X_train, y_train)


# 분석 모델 객체의 평가(R2 스코어 확인 - 결정계수)
print("LR 평가 : ", lr_model.score(X_train, y_train))
print("Ridge 평가 : ", ridge_model.score(X_train, y_train))
print("Lasso 평가 : ", lasso_model.score(X_train, y_train))

from matplotlib import pyplot as plt

coef_range = list(range(1, len(ridge_model.coef_) + 1))

plt.plot(coef_range, lr_model.coef_, 'r^')
plt.plot(coef_range, ridge_model.coef_, 'bo')
plt.plot(coef_range, lasso_model.coef_, 'gv')

plt.hlines(0, 1, len(ridge_model.coef_) + 1, 
           colors='y', linestyles='dashed')

plt.show()



















