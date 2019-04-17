# -*- coding: utf-8 -*-

# numpy는 과학 계산을 위한 라이브러리
# 다차원 배열을 처리하는데 필요한 여러 유용한 기능을 제공

# 설치 명령
# pip install numpy

import numpy as np

# numpy 배열 생성 방법
# 파이썬의 리스트를 활용한 생성
# np.array(파이썬 리스트/변수)

python_list = [1, 2, 3, 4]
np_array_1 = np.array(python_list)
print(f"np_array_1 -> {np_array_1}")
print(f"type(np_array_1) -> {type(np_array_1)}")
print(f"np_array_1.shape -> {np_array_1.shape}")

# numpy 배열의 인덱스 연산 방법
# 시작 인덱스는 0으로 시작하면
# 종료 인덱스는 -1 을 사용할 수 있음
print(f"np_array_1[0] -> {np_array_1[0]}")
print(f"np_array_1[-1] -> {np_array_1[-1]}")

# 배열의 크기 값을 확인하는 방법
print(f"len(np_array_1) -> {len(np_array_1)}")
print(f"np_array_1.shape[0] -> {np_array_1.shape[0]}")
 
# numpy의 다차원 배열을 생성하는 코드
np_array_2 = np.array([[1,2,3],[4,5,6]])
print(f"np_array_2 -> {np_array_2}")
print(f"type(np_array_2) -> {type(np_array_2)}")
print(f"np_array_2.shape -> {np_array_2.shape}")

# numpy 다차원 배열의 요소에 접근하는 인덱스 연산
# 배열명[행, 열] : 인덱스의 시작은 0
print(f"np_array_2[0,0] -> {np_array_2[0,0]}")
print(f"np_array_2[-1,-1] -> {np_array_2[-1,-1]}")

# 다차원 배열의 len 함수 결과
# 이차원 배열의 이름을 사용하는 경우 행의 값이 반환
print(f"len(np_array_2) -> {len(np_array_2)}")
# 이차원 배열의 행의 인덱스를 사용하는 경우 
# 해당 행의 열의 개수가 반환
print(f"len(np_array_2[0]) -> {len(np_array_2[0])}")

# shape 속성의 값을 사용하여 길이를 반환받는 방법
# 인덱스의 값을 0 으로 사용하는 경우 행의 값이 반환
# 인덱스의 값을 1 으로 지정하는 경우 열의 값이 반환
print(f"np_array_2.shape[0] -> {np_array_2.shape[0]}")
print(f"np_array_2.shape[1] -> {np_array_2.shape[1]}")


























