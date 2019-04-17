# -*- coding: utf-8 -*-

import numpy as np

python_list = list(range(1,11))

numpy_array_1 = np.array(python_list)
print(f"numpy_array_1.shape -> {numpy_array_1.shape}")
print(f"numpy_array_1 -> {numpy_array_1}")

# 배열의 전치 연산을 수행할 수 있는 np.transpose 함수
# 1차원인 배열의 경우 변함이 없음
numpy_array_2 = np.transpose(numpy_array_1)
print(f"numpy_array_2.shape -> {numpy_array_2.shape}")
print(f"numpy_array_2 -> {numpy_array_2}")

# 배열의 전치 연산을 수행할 수 있는 np.transpose 함수
# 2차원인 배열의 경우 행과 열이 변환
# (전치행열의 경우 차원의 수는 유지됨)
numpy_array_1_reshape = numpy_array_1.reshape(-1,1)
print(f"numpy_array_1_reshape -> {numpy_array_1_reshape}")

numpy_array_3 = np.transpose(numpy_array_1_reshape)
print(f"numpy_array_3.shape -> {numpy_array_3.shape}")
print(f"numpy_array_3 -> {numpy_array_3}")

numpy_array_4 = numpy_array_1_reshape.reshape(-1)
print(f"numpy_array_4.shape -> {numpy_array_4.shape}")
print(f"numpy_array_4 -> {numpy_array_4}")


















