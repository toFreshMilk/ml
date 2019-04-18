# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

iris = load_iris()
model = LogisticRegression(solver='liblinear', multi_class='ovr')

print("Iris 레이블:\n{}".format(iris.target))

scores = cross_val_score(model, iris.data, iris.target, cv=3)
print("교차 검증 점수(cv 3) : {}".format(scores))

from sklearn.model_selection import KFold

#그냥 셔플이 아닌 비율을 맞춘 셔플
kfold = KFold(n_splits=3, shuffle=True, random_state=0)
scores = cross_val_score(model, iris.data, iris.target, cv=kfold)
print("교차 검증 점수(KFold 3) : {}".format(scores))









