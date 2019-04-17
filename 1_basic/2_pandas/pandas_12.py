# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# 데이터프레임의 병합
# concat 함수의 사용
# 단순 데이터 연결을 지원
# 수평으로 데이터를 연결하는 경우 axis=1로 인수를 설정

df1 = pd.DataFrame(
    np.arange(6).reshape(-1, 2),
    index=['a', 'b', 'c'],
    columns=['데이터1', '데이터2'])

print(df1)

df2 = pd.DataFrame(
    5 + np.arange(4).reshape(2, 2),
    index=['a', 'c'],
    columns=['데이터3', '데이터4'])

print(df2)

print(pd.concat([df1, df2], axis=1))








