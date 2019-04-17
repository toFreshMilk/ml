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

# 특정 열에 존재하는 결측 데이터를 임의의 값으로
# 설정하기 위한 변환기
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameChangeNaN (BaseEstimator, TransformerMixin) :
    def __init__(self, change_value, attr_name=None) :
        self.change_value = change_value
        self.attr_name = attr_name
        
    def fit(self, X, y=None) :
        return self
    
    def transform(self, X) :
        if self.attr_name is not None :
            #X[self.attr_name].fillna(self.change_value, inplace=True)
            mean_v = X[self.attr_name].mean()
            X[self.attr_name].fillna(mean_v, inplace=True)
        else :
            X.fillna(self.change_value, inplace=True)
            
        # 결측데이터가 임의의 값으로 변환된 
        # numpy 배열을 반환
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
print(diabetes.describe())

#changeNaN = DataFrameChangeNaN(0.3)
changeNaN = DataFrameChangeNaN(0.3, attr_name='A')

changeNaN.fit(diabetes)
print(changeNaN.transform(diabetes))

#print(changeNaN.fit_transform(diabetes))




















