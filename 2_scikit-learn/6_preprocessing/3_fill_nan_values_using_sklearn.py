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

# 결측 데이터
# 각각의 샘플내에 포함된 특성의 값이 존재하지 않은 경우를 의미
# 결측 데이터가 존재하는 경우 학습이 원활하게 진행되지 않기 때문에
# 머신러닝 모델의 학습 전에 반드시 결측 데이터를 해결해야함

import numpy as np
import pandas as pd

diabetes = pd.read_csv("../../data/diabetes.csv", header=None)
diabetes.columns = ['A','B','C','D','E','F','G','H','I']
print(diabetes.info())

# 1행 1열과 2행 1열에 None(np.NaN)값을 대입
diabetes.iloc[[0, 1], 0] = None
# 결측 데이터가 존재하는 경우 데이터 샘플의 수가 전체 데이터 샘플의
# 수와 다름을 확인할 수 있음
print(diabetes.info())
# 결측 데이터는 NaN 값으로 확인됨
print(diabetes.head())

# 사이킷 런의 변환기 클래스
# - 학습 및 테스트 데이터를 수정할 수 있는 기능을 제공
# - 결측 데이터의 수정, 스케일 변환, 특성 추가 등과 같은 기능을 제공

# sklearn.impute.SimpleImputer 변환기 클래스
# 데이터 내부에 존재하는 모든 결측 데이터에 대해서
# 중간 값, 평균 값, 최빈도 값, 사용자 지정 값으로 
# 수정할 수 있는 기능을 제공하는 클래스

# 사이킷 런의 모든 예측기 클래스들은
# fit, transform, fit_transform(클래스에 따라 제공되지 않을 수도 있음) 
# 메소드를 제공
from sklearn.impute import SimpleImputer

# 결측 데이터를 해당 컬럼의 중간값으로 대체할 수 있는
# 사이킷런 변환 클래스의 객체 생성
imputer = SimpleImputer(strategy="median")
# 결측 데이터를 해당 컬럼의 평균값으로 대체할 수 있는
# 사이킷런 변환 클래스의 객체 생성
#imputer = SimpleImputer(strategy="mean")
# 결측 데이터를 해당 컬럼의 가장 빈도수가 높은 값으로 
# 대체할 수 있는 사이킷런 변환 클래스의 객체 생성
#imputer = SimpleImputer(strategy="most_frequent")
# 결측 데이터를 fill_value 매개변수에 지정한 값으로
# 대체할 수 있는 사이킷런 변환 클래스의 객체 생성
#imputer = SimpleImputer(strategy="constant", fill_value=0.25)

print(imputer.fit_transform(diabetes)[:5])

# 변환기의 결과를 사용하여 데이터프레임 객체를 새롭게 생성
diabetes = pd.DataFrame(imputer.fit_transform(diabetes))

# 결측데이터가 삭제된 것을 확인할 수 있음
print(diabetes.info())
print(diabetes.head())




















