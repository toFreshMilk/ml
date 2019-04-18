# -*- coding: utf-8 -*-

# 데이터전처리
# 데이터 분석을 위한 데이터 처리 과정
# - 전체 데이터 셋에서 데이터 분석에 사용될 열 선정
# - 특정 열에 존재하는 빈 값을 제거하거나
#   또는 특정 열에 존재하는 빈 값을 임의의 값으로 변경
# - 데이터의 스케일(값의 범위) 조정
# - 범주형 변수의 값 변경
#   (문자열 값의 수치 데이터화)
#   (원핫인코딩 처리)
# - 학습, 테스트 데이터 분할

# 특성 데이터의 스케일링
# 각각의 특성 데이터들은 고유의 특성과 분포를 가지고 있음
# 각각의 특성 데이터의 값을 그대로 사용하게 되면 
# 학습 속도가 저하되거나 학습이 되지 않는 문제가 발생할 수 있음
# (최근접 이웃 알고리즘 및 SVM 알고리즘)
# 이러한 경우 사이킷 런의 Scaler 클래스를 이용하여 
# 각각의 특성 데이터들을 일정 범위로 스케일링할 수 있음

#from sklearn.preprocessing import MinMaxScaler
#data = [[-1,2], 
#        [-0.5,6], 
#        [0,10], 
#        [1,18]]
#scaler = MinMaxScaler()
#print(scaler.fit_transform(data))

import pandas as pd

data = pd.read_csv("../../data/score.csv")
data.drop("name", axis=1, inplace=True)

# 각 수치데이터의 최소 최대값을 기준으로 정규화를
# 처리할 수 있는 MinMaxScaler 클래스
# - 각 열의 데이터를 0 ~ 1 사이로 압축하는 역할
# - 반드시 수치 데이터만을 전달해야 함
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
print(scaler.fit_transform(data))

print("=" * 20)

# 각 수치데이터들을 평균을 0, 분산을 1로 하는 
# 표준정규분포를 따르는 값으로 변환시킬 수 있는 
# StandardScaler 클래스
# - 각 열의 데이터들이 최소값과 최대값이 한정되지 않음
# - 일반적으로 StandardScaler 정규화를 처리하는 것이
#   데이터 분석 성능이 높아짐
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print(scaler.fit_transform(data))

print("=" * 20)

# 중앙값(median)과 IQR(interquartile range)를 사용하여 
# 아웃라이어의 영향을 최소화하며 변환할 수 있는 
# RobustScaler 클래스
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
print(scaler.fit_transform(data))



















