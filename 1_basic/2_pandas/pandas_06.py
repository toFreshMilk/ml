# -*- coding: utf-8 -*-

import pandas as pd

data = {
    "year" : [2017, 2018, 2019],
    "GDP Rate" : [2.8, 3.1, 3.0], 
    "GDP" : ['1.637M', '1.859M', '2.237M']
}

df = pd.DataFrame(data)

# 데이터프레임의 앞 부분의 데이터 추출
# 기본값 = 5개
print(df.head(2))

# 데이터프레임의 뒷 부분의 데이터 추출
# 기본값 = 5개
print(df.tail())








