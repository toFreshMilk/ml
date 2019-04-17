# -*- coding: utf-8 -*-

import numpy as np
 
# numpy bool 인덱싱
# 배열 각 요소의 선택 여부를 True, False로 표현하는 방식

python_list = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

numpy_array = np.array(python_list)
 
# numpy 배열에 대한 연산의 결과를 사용한 
# bool 인덱싱 처리 방법
bool_indexing_array = numpy_array % 2 == 0
print(bool_indexing_array)
 
bool_indexing_result = numpy_array[bool_indexing_array];
print(bool_indexing_result)    

# numpy 배열의 인덱스에 bool 인덱싱 처리 방법
bool_indexing_result = numpy_array[numpy_array % 2 == 0]
print(bool_indexing_result)    
















