# -*- coding: utf-8 -*-

import numpy as np

# numpy 배열 생성 방법
# numpy 모듈의 함수를 사용하여 배열 생성
# np.zeros(배열의 형태/크기) : 모든 요소를 0으로 초기화 하여 
# 지정된 크기의 배열 생성
numpy_array_1 = np.zeros((2,2))
print(f"zeros -> \n {numpy_array_1}")

# np.ones(배열의 형태/크기) : 모든 요소를 1로 초기화 하여 
# 지정된 크기의 배열 생성 
numpy_array_2 = np.ones((2,3))
print(f"ones -> \n {numpy_array_2}")

# np.full(배열의 형태/크기, 값) : 모든 요소를 지정된 값으로 초기화 하여 
# 지정된 크기의 배열 생성
numpy_array_3 = np.full((2,3), 5)
print(f"full -> \n {numpy_array_3}")

# np.eye(배열의 행) : 대각선에 해당되는 모든 요소를 1로 채우고 
# 나머지 요소들은 0으로 초기화 하여 지정된 크기의 2차원 배열 생성
numpy_array_4 = np.eye(10)
print(f"eye -> \n {numpy_array_4}")







