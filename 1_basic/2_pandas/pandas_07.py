# -*- coding: utf-8 -*-

import pandas as pd

# 데이터프레임의 병합
# merge 함수의 사용

df1 = pd.DataFrame({
    '고객번호': [1001, 1002, 1003, 1004, 1005, 1006, 1007],
    '이름': ['둘리', '도우너', '또치', '길동', '희동', '마이콜', '영희']
})

print(df1)
print("=" * 15)

df2 = pd.DataFrame({
    '고객번호': [1001, 1001, 1005, 1006, 1008, 1001],
    '금액': [10000, 20000, 15000, 5000, 100000, 30000]
})

print(df2)
print("=" * 15)

# pandas.merge 
# 동일한 컬럼 정보를 가진 두개의 데이터프레임을
# 동일한 컬럼 정보를 기준으로 병합
# 기본적으로  INNER JOIN을 실행
# (동일한 값이 존재하는 경우만 병합)
print(pd.merge(df1, df2))
print("=" * 15)

# merge 함수의 how 매개변수를 outer로 지정하는 경우
# outter 조인으로 병합
print(pd.merge(df1, df2, how='outer'))
print("=" * 15)

# left, right outer 조인을 지원
print(pd.merge(df1, df2, how='left'))
print("=" * 15)

print(pd.merge(df1, df2, how='right'))
print("=" * 15)










