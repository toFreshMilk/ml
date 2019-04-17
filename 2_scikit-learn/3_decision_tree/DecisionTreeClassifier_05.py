# -*- coding: utf-8 -*-

import numpy as np

X = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1, 1)
y = np.array([10,20,30,40,50,60,70,80,90,100])

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

knn_model = KNeighborsRegressor(n_neighbors=3).fit(X, y)
lr_model = LinearRegression().fit(X, y)
dt_model = DecisionTreeRegressor().fit(X, y)

print("훈련 정확도(KNN): {:.3f}".format(knn_model.score(X, y)))
print("훈련 정확도(LR): {:.3f}".format(lr_model.score(X, y)))
print("훈련 정확도(DT): {:.3f}".format(dt_model.score(X, y)))

# 결정 트리 또는 최근접 이웃 알고리즘을 활용하는 경우의 주의사항
# 학습에 사용된 특성 데이터(X)의 범주를 벗어나느 데이터를 사용하여
# 예측하려는 경우 선형 모델과 다르게 학습 데이터의 영역을 벗어난 
# 값을 예측할 수 없습니다.

# 시계열 데이터와 같은 경우 되도록 선형 모델을 활용하여
# 예측해야 합니다.

# 학습엥 사용된 X의 최대값 10을 넘어가는 데이터를 예측하려는 경우
# 결정 트리는 학습 데이터에서 사용된 y의 최대값(10) 만을 반환합니다.
X_test = np.array([100]).reshape(-1, 1)

print("100에 대한 예측 결과(KNN) : ", knn_model.predict(X_test))
print("100에 대한 예측 결과(LR) : ", lr_model.predict(X_test))
print("100에 대한 예측 결과(DT) : ", dt_model.predict(X_test))

from sklearn.tree import export_graphviz

export_graphviz(dt_model, out_file='error_tree.dot',                
                feature_names=["X1"], filled=True)

import graphviz
from IPython.display import display

with open('error_tree.dot', encoding='utf-8') as f:
    dot_graph = f.read()
    
display(graphviz.Source(dot_graph))














