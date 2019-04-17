# -*- coding: utf-8 -*-

import numpy as np
 
# numpy 배열의 인덱싱 연산
python_list = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
]

numpy_array = np.array(python_list)

# 아래의 코드는 numpy_array 의 
# (0, 1), (2, 3) 요소를 추출하는 코드
# numpy_array[
# [행의 인덱스 리스트], [열의 인덱스 리스트]] 
indexing = numpy_array[[0, 2], [1, 3]] 
print(indexing)








