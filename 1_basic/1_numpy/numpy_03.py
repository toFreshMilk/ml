# -*- coding: utf-8 -*-

import numpy as np

python_list = list(range(1,11))

numpy_array_1 = np.array(python_list)
print(f"numpy_array_1.shape -> {numpy_array_1.shape}")
print(f"numpy_array_1 -> {numpy_array_1}")

# 배열의 형태를 수정할 수 있는 reshape() 메소드
# 1차원 배열을 2차원 배열로 형태를 변환하는 예제
numpy_array_2 = numpy_array_1.reshape(-1, 2)
print(f"numpy_array_1.shape -> {numpy_array_1.shape}")
print(f"numpy_array_2.shape -> {numpy_array_2.shape}")
print(f"numpy_array_2 -> {numpy_array_2}")

# 2차원 배열을 1차원 배열로 형태를 변환하는 예제
numpy_array_3 = numpy_array_2.reshape(-1)
print(f"numpy_array_3.shape -> {numpy_array_3.shape}")
print(f"numpy_array_3 -> {numpy_array_3}")










