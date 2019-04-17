# -*- coding: utf-8 -*-

import pandas as pd

data = {
    "year" : [2017, 2018, 2019],
    "GDP Rate" : [2.8, 3.1, 3.0], 
    "GDP" : ['1.637M', '1.859M', '2.237M']
}

df = pd.DataFrame(data)

# year 컬럼의 데이터 중 
# 2018년도 이후인 데이터를 추출
print( df.year >= 2018 )
print( df[df.year >= 2018]  )

# 특정 조건에 맞는 일부분의 컬럼 데이터 조회
# 데이터프레임변수[열의이름][조건식]
print("=" * 17)
print( df['GDP'][df.year >= 2018] )
print("=" * 17)
print( df[ ['GDP','GDP Rate'] ][df.year >= 2018] )












