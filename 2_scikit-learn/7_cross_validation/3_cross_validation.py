# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

iris = load_iris()
model = LogisticRegression(solver='liblinear', multi_class='ovr')

print("Iris 레이블:\n{}".format(iris.target))


#적절히 섞어준다. 무작위로.
scores = cross_val_score(model, iris.data, iris.target, cv=3)
print("교차 검증 점수(cv 3) : {}".format(scores))


#무작위가 아닌 섞음변수를 준다.
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5)
scores = cross_val_score(model, iris.data, iris.target, cv=kfold)
print("교차 검증 점수(KFold 5) : {}".format(scores))

kfold = KFold(n_splits=3)
scores = cross_val_score(model, iris.data, iris.target, cv=kfold)
print("교차 검증 점수(KFold 3) : {}".format(scores))









