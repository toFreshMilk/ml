# -*- coding: utf-8 -*-

import pandas as pd

# 파일의 경로를 포함한 이름을 저장
fname = '../../data/iris.csv'
# 특정 파일을 DataFrame으로 로딩
df_iris = pd.read_csv(fname)

# iris 데이터의 일부분을 확인
print(df_iris.head())
print(df_iris.tail())

# iris 데이터의 개수 및 결측데이터 확인
print(df_iris.info())

# iris 데이터의 수치 데이터 확인
print(df_iris.describe())

# iris 데이터는 iris 품종을 맞추기 위한 
# 데이터 셋으로 앞의 4가지가 특성 데이터
# 마지막 열의 데이터가 라벨 데이터임

# 특성 데이터와 라벨 데이터를 분할하는 작업
# DataFrame의 iloc 연산
# 인덱스 정보를 기반으로 데이터프레임을 분할
# 아래의 코드는 전체 행(샘플)에서 마지막 열을
# 제외하고 분할하는 코드
X_df = df_iris.iloc[:, :-1]
# 아래의 코드는 전체 행(샘플)에서 마지막 열을
# 추출하는 코드
y_df = df_iris.iloc[:, -1]

# 라벨 데이터의 분포를 확인
print(type(y_df))
print(y_df.value_counts())
print(y_df.value_counts() / len(y_df))

# 특정 데이터의 분포를 확인
from matplotlib import pyplot as plt
X_df.hist()
plt.show()

X_df.plot.hist()
plt.show()



















