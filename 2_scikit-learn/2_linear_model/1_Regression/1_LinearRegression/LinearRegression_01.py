# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

# 피자 크기에 따른 가격 데이터
X = [6, 8, 10, 14, 18]
y = [7, 9, 13, 17.5, 18]

plt.figure()
plt.title('Pizza price plotted against diameter')
plt.xlabel('Diameter in inches')
plt.ylabel('Price in dollars')
plt.plot(X, y, 'ko')
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.show()