# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.datasets import load_boston

boston = load_boston()

print(boston.keys())

# 특성 데이터의 데이터프레임 생성
X_df = pd.DataFrame(boston.data)
X_df.columns = boston.feature_names

# 라벨 데이터의 데이터프레임 생성
y_df = pd.DataFrame(boston.target)

# 특성 데이터의 샘플 개수 및 결측 데이터 확인
print(X_df.info())

pd.options.display.max_columns = 100

# 특성 데이터의 수치데이터 확인
print(X_df.describe())

# 라벨 데이터 확인
print(y_df.head())
print(y_df.describe())

# 라벨 데이터가 수치 데이터인 경우
# (회기분석인 경우)
# 반드시 값의 분포를 확인
# 양쪽 끝단에 데이터가 치우치고 있는 지
# 확인
from matplotlib import pyplot as plt
y_df.hist()
plt.show()

# 특성 데이터와 라벨 데이터의 결합
all_df = pd.concat([X_df, y_df], axis=1)

# 상관관계를 확인
corr_df = all_df.corr()
print(corr_df)

# 데이터 분석
from sklearn.svm import SVR
model = SVR()
model.fit(X_df.values, y_df.values.reshape(-1))

print("학습 결과", 
      model.score(X_df.values, 
                  y_df.values.reshape(-1)))
















