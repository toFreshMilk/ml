# -*- coding: utf-8 -*-

import pandas as pd

data = {
    "year" : [2017, 2018, 2019],
    "GDP Rate" : [2.8, 3.1, 3.0], 
    "GDP" : ['1.637M', '1.859M', '2.237M']
}

df = pd.DataFrame(data)

# 각 열의 데이터에 접근하기 위해서
# 열의 이름을 사용
# 데이터프레임변수명.열의이름
print(df.year)
print(df.GDP)

# 만약 열의 이름에 공백이 존재하는 경우
# [] 연산을 사용하여 접근
# 데이터프레임변수명["열의이름"]
print(df['GDP Rate'])
















