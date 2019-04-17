# -*- coding: utf-8 -*-

import pandas as pd

# 데이터프레임의 병합
# concat 함수의 사용
# 단순 데이터 연결을 지원
# 기본 연산은 데이터프레임을 상하로 결합

s1 = pd.Series([0, 1], index=['A', 'B'])

print(s1)

s2 = pd.Series([2, 3, 4], index=['A', 'B', 'C'])

print(s2)

print(pd.concat([s1, s2]))
