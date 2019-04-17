# -*- coding: utf-8 -*-

import numpy as np
 
# numpy 배열의 MATRIX PRODUCT(행렬 곱)
# np.dot 함수를 사용하여 계산
# 앞의 배열의 열의 수와 뒤의 배열의 행의 수가
# 일치해야만 연산이 성립됨
python_list_1 = [
    [1,2],
    [3,4]
]
python_list_2 = [
    [5,6],
    [7,8]
]

numpy_array_1 = np.array(python_list_1)
numpy_array_2 = np.array(python_list_2)
 
numpy_array_r = np.dot(numpy_array_1, numpy_array_2)
print(numpy_array_r)










