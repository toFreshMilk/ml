# -*- coding: utf-8 -*-

import pandas as pd

fname = '../../data/score.csv'
df = pd.read_csv(fname)

df.drop('name', axis=1, inplace=True)

X_df = df.iloc[:, 1:]
y_df = df.iloc[:, 0]

X_train = X_df.values
y_train = y_df.values

from sklearn.neighbors import KNeighborsRegressor

# 선형 모델에 L2 제약 조건을 추가한 Ridge 클래스
# L2 제약 조건 : 모든 특성에 대한 가중치의 값을
# 0 주변으로 위치하도록 제어하는 제약조건
# LinearRegression 클래스는 학습 데이터에 최적화되도록
# 학습을 하기때문에 테스트 데이터에 대한 일반화 성능이 감소됩니다.
# 이러한 경우 모든 특정 데이터를 적절히 활용할 수 있도록
# L2 제약 조건을 사용할 수 있으며, L2 제약조건으로 인하여
# 모델의 일반화 성능이 증가하게 됩니다.
from sklearn.linear_model import LinearRegression, Ridge

# 분석 모델 클래스의 객체 생성
knn_model = KNeighborsRegressor(n_neighbors=2).fit(X_train, y_train)
lr_model = LinearRegression().fit(X_train, y_train)

# Ridge 클래스의 하이퍼 파라메터 alpha
# alpha의 값이 커질수록 제약을 크게 설정
# (alpha의 값이 커질수돌 모든 특성들의 가중치의 값은 0 주변으로 위치함)
# alpha의 값이 작아질수록 제약이 약해짐
# (alpha의 값이 작아질수록 모든 특성들의 가중치의 값은 0에서 멀어짐)
ridge_model = Ridge(alpha=100).fit(X_train, y_train)


# 분석 모델 객체의 평가(R2 스코어 확인 - 결정계수)
print("KNN 평가 : ", knn_model.score(X_train, y_train))
print("LR 평가 : ", lr_model.score(X_train, y_train))
print("Ridge 평가 : ", ridge_model.score(X_train, y_train))

from matplotlib import pyplot as plt

coef_range = list(range(1, len(ridge_model.coef_) + 1))

plt.plot(coef_range, lr_model.coef_, 'r^')
plt.plot(coef_range, ridge_model.coef_, 'bo')

plt.hlines(0, 1, len(ridge_model.coef_) + 1, 
           colors='y', linestyles='dashed')

plt.show()



















