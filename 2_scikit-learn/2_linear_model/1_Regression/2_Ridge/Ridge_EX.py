# -*- coding: utf-8 -*-

# 사이킷 런의 load_diabetes 함수를 사용하여 당료병 수치를 예측할 수 있는 모델을
# 작성한 후 테스트하세요.(LinearRegression, Ridge 클래스를 활용)
# - Ridge 클래스의 alpha 값을 조절하여 값의 변화를 확인하세요.

import pandas as pd
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()

X_df = pd.DataFrame(diabetes.data)
y_df = pd.Series(diabetes.target)

pd.options.display.max_columns = 100
print(X_df.describe())
print(y_df.describe())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X_df.values, y_df.values, random_state=1)
    
# 데이터 전처리 과정 추가
from sklearn.preprocessing import PolynomialFeatures
# 원본 특성데이터를 3차 방정식의 형태로 가공하는 
# PolynomialFeatures 객체를 생성
poly = PolynomialFeatures(degree=3, include_bias=False)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
    
from sklearn.linear_model import LinearRegression, Ridge

lr_model = LinearRegression().fit(X_train_poly, y_train)
ridge_model = Ridge(alpha=0.001).fit(X_train_poly, y_train)

print("학습 데이터 평가(LR) : ", lr_model.score(X_train_poly, y_train))
print("학습 데이터 평가(Ridge) : ", ridge_model.score(X_train_poly, y_train))

print("=" * 35)

print("테스트 데이터 평가(LR) : ", lr_model.score(X_test_poly, y_test))
print("테스트 데이터 평가(Ridge) : ", ridge_model.score(X_test_poly, y_test))

from matplotlib import pyplot as plt

coef_range = list(range(1, len(ridge_model.coef_) + 1))

plt.plot(coef_range, lr_model.coef_, 'r^')
plt.plot(coef_range, ridge_model.coef_, 'bo')

plt.hlines(0, 1, len(ridge_model.coef_) + 1, 
           colors='y', linestyles='dashed')

plt.show()
















