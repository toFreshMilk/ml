# -*- coding: utf-8 -*-

# data 폴데에 있는 score.csv 파일을 읽어오세요.
# iq,academy,game,tv 점수를 X 데이터로 활용하여
# score 점수(y)를 예측할 수 있는 회기 모델을 테스트하세요
# r2 score를 출력

import pandas as pd

fname = '../../../data/score.csv'
df = pd.read_csv(fname)

# pandas의 DataFrame에서 특정 열을 삭제하는 방법
# drop 메소드를 사용
# 데이터프레임변수명.drop(컬럼명, axis=1)
# 동작하는 방식 : 매개변수로 전달된 컬럼명의 열을
# 삭제한 데이터프레임을 반환
# (원본 데이터프레임은 변화가 없음)
# df.drop('name', axis=1)

# 만약 원본 데이터프레임에서 해당 열을 삭제하려는
# 경우 inplace=True 매개변수를 전달합니다.
df.drop('name', axis=1, inplace=True)
print(df)

# X_df -> DataFrame
X_df = df.iloc[:, 1:]
print(type(X_df))

# y_df -> Series
y_df = df.iloc[:, 0]
print(type(y_df))

# DataFrame의 모든 데이터를 numpy 배열로 변환
X_train = X_df.values
# Series 모든 데이터를 numpy 배열로 변환
y_train = y_df.values

# 분석 모델 클래스의 import
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

# 분석 모델 클래스의 객체 생성
knn_model = KNeighborsRegressor(n_neighbors=2)
lr_model = LinearRegression()

# 분석 모델 객체의 학습
knn_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

# 분석 모델 객체의 평가(R2 스코어 확인 - 결정계수)
print("KNN 평가 : ", knn_model.score(X_train, y_train))
print("LR 평가 : ", lr_model.score(X_train, y_train))

















