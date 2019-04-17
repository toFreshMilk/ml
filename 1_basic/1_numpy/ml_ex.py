# -*- coding: utf-8 -*-

import numpy as np
from sklearn.linear_model import LinearRegression

# 샘플데이터 및 타겟데이터 생성
X = np.array([10,20,30,40,50])
y = np.array([97,202,301,397,505])

# 머신러닝 모델의 객체 생성
model = LinearRegression()

# 머신러닝 모델의 학습 진행
model.fit(X.reshape(-1,1), y)

pred = model.predict(X.reshape(-1,1))
print("예측결과 : ", pred)
print("예측성능 : ", model.score(X.reshape(-1,1), y))

from matplotlib import pyplot as plt

plt.scatter(X, y)
plt.plot(X, pred)
plt.show()




















