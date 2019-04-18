# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

iris = load_iris()
logreg = LogisticRegression(solver='liblinear', multi_class='ovr')

#데이터 3분할 CV.
scores = cross_val_score(logreg, iris.data, iris.target, cv=2)

print("교차 검증 점수 : {}".format(scores))

print("교차 검증 평균 점수 : {:.2f}".format(scores.mean()))


#