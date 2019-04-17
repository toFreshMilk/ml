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

# 사이킷런의 변환기 클래스들은 상속을 기반으로
# 동작하지 않습니다. (덕 타이핑 기반으로 동작하는 방식을 사용)
# - 덕 타이핑 : 상속이나 인터페이스 구현이 아니라 
#   객체의 속성이나 메서드가 객체의 유형을 결정하는 방식
# 변환기로 동작하기 위한 클래스들은 
# fit, transform, fit_transform 메소드를 정의해야 합니다.
# (fit_transform 메소드를 정의하지 않은 경우
# fit 메소드의 호출 후 transform 메소드가 자동호출)

# 특정 열에 존재하는 결측데이터를 제거하기 위한 변환기

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameDropNaN (BaseEstimator, TransformerMixin) :
    def __init__(self) :
        pass
    
    def fit(self, X, y=None) :
        return self
    
    def transform(self, X) :
        # 결측데이터를 제거하기 위한 pandas 메소드
        # dropna 메소드를 사용하여 NaN 데이터 삭제
        # dropna 메소드의 axis 값을 0 또는 1로 지정
        # 0 : 결측데이터가 포함된 행을 제거
        # 1 : 결측데이터가 포함된 열을 제거
        # inplace 매개변수를 True 로 지정하면
        # 실제 삭제된 결과는 해당 데이터프레임에 
        # 적용하고 어떤값도 반환하지 않습니다.
        X.dropna(axis=0, inplace=True)
        # 결측데이터가 제거된 numpy 배열을 반환
        return X.values

diabetes = pd.read_csv("../../data/diabetes.csv", header=None)
diabetes.columns = ['A','B','C','D','E','F','G','H','I']
print(diabetes.info())

# 1행 1열과 2행 1열에 None 값 대입
diabetes.iloc[[0, 1], 0] = None

# 3행 2열에 None 값 대입
import numpy as np
diabetes.iloc[2, 1] = np.nan
print(diabetes.info())
print(diabetes.head())

dropNaN = DataFrameDropNaN()
dropNaN.fit(diabetes)
print(dropNaN.transform(diabetes))
print(dropNaN.transform(diabetes).shape)


















