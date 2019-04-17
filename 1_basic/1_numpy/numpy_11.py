# -*- coding: utf-8 -*-

import numpy as np
 
numpy_array = np.array([[1,2],[3,4]])
print(numpy_array)

result = np.sum(numpy_array)
print(result)
 
# axis = 0, 컬럼
# axis = 1, 행
result = np.sum(numpy_array, axis=0)
print(result)
 
result = np.sum(numpy_array, axis=1)
print(result)

result = np.prod(numpy_array)
print(result)

result = np.prod(numpy_array, axis=0)
print(result)

result = np.prod(numpy_array, axis=1)
print(result)



