# -*- coding: utf-8 -*-

import numpy as np
 
numpy_array_1 = np.array([1, 2, 3])
numpy_array_2 = np.array([4, 5, 6])

# numpy 배열을 왼쪽에서 오른쪽으로 결합
r = np.r_[numpy_array_1, numpy_array_2]
print(r)

# numpy 배열을 왼쪽에서 오른쪽으로 결합
r = np.hstack([numpy_array_1, numpy_array_2])
print(r)

# 2개의 1차원 numpy 배열을 
# 세로로 결합하여 2차원 배열 생성
# (각 열의 결합하여 2차원 배열을 생성)
r = np.c_[numpy_array_1, numpy_array_2]
print(r)

# numpy 배열을 왼쪽에서 오른쪽으로 결합
# 세로로 결합하여 2차원 배열 생성
r = np.column_stack([numpy_array_1, numpy_array_2])
print(r)









