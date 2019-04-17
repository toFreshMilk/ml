# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 20:08:49 2019

@author: 502-22
"""

import numpy as np

#행은 맘대로, 열은 1 고정
X = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1, 1)
y = np.array([10,20,30,40,50,60,70,80,90,100])

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


knn_model = KNeighborsRegressor(n_neighbors=3).fit(X,y)
lr_model = LinearRegression().fit(X,y)
dt_model = DecisionTreeRegressor().fit(X, y)


print("훈련 정확도(KNN): {:.3f}".format(knn_model.score(X, y)))

print("훈련 정확도(LR): {:.3f}".format(lr_model.score(X, y)))

print("훈련 정확도(DT): {:.3f}".format(dt_model.score(X, y)))


#LR과 DT의 단점 - 샘플으 ㅣ최소와 최대로밖에 반응 못함
