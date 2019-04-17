# -*- coding: utf-8 -*-

import pandas as pd

# 데이터프레임의 병합
# merge 함수의 사용

df1 = pd.DataFrame({
    '품종': ['setosa', 'setosa', 'virginica', 'virginica'],
    '꽃잎길이': [1.4, 1.3, 1.5, 1.3]})

df2 = pd.DataFrame({
    '품종': ['setosa', 'virginica', 'virginica', 'versicolor'],
    '꽃잎너비': [0.4, 0.3, 0.5, 0.3]})

# 두개의 데이터 프레임에 동일한 키값들이 존재하는 경우
# 생성 가능한 모든 조합을 반환
print(pd.merge(df1, df2))