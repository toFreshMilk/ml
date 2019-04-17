# -*- coding: utf-8 -*-

import pandas as pd

# pandas의 데이터 구조
# 1차원 : Series
# 2차원 : DataFrame
# 3차원 : Panel

data = {
    "year" : [2017, 2018, 2019],
    "GDP Rate" : [2.8, 3.1, 3.0], 
    "GDP" : ['1.637M', '1.859M', '2.237M']
}

df = pd.DataFrame(data)

print(df)
print(f"type(df) -> {type(df)}")














