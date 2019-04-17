# -*- coding: utf-8 -*-

import numpy as np

X = np.array([6, 8, 10, 14, 18]).reshape(-1,1)
y = np.array([7, 9, 13, 17.5, 18])

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X, y)

print("학습데이터의 평가점수 : ", model.score(X,y))

X_test = np.array(list(range(3,21))).reshape(-1,1)
pred = model.predict(X_test)

from matplotlib import pyplot as plt

plt.figure()
plt.title('Pizza price plotted against diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.plot(X, y, 'ko')
plt.axis([0, 25, 0, 25])
plt.grid(True)

plt.plot(X_test, pred, 'r--')

plt.show()



























