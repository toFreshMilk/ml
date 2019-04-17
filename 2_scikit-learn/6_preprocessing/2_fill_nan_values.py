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

# 결측 데이터를 임의의 값으로 설정하는 방법 
# - pandas를 사용하여 결측 데이터를 임의의 값으로 설정하는 방법

# 결측데이터를 제거하기 위한 pandas 메소드
# fillna 메소드를 사용하여 NaN 데이터를 임의의 값으로 설정
# inplace 매개변수를 True 로 지정하면
# 실제 수정된 결과를 해당 데이터프레임에 
# 적용하고 어떤값도 반환하지 않습니다.
mean_A = diabetes.A.mean()
diabetes.A.fillna(mean_A, inplace=True)        

# 결측데이터가 삭제된 것을 확인할 수 있음
print(diabetes.info())
print(diabetes.head())




















