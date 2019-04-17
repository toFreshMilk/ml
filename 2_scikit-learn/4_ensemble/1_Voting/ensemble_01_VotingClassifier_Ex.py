# -*- coding: utf-8 -*-

# winequality-red.csv 데이터를 LogisticRegression, KNeighborsClassifier, 
# DecisionTreeClassifier를 조합한 VotingClassifier로 분석한 후, 
# 결과를 확인하세요.

import pandas as pd
from matplotlib import pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

fname = '../../data/winequality-red.csv'
df = pd.read_csv(fname, sep=';')

X_df = df.iloc[:, :-1]
y_df = df.iloc[:, -1]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
    train_test_split(X_df.values, y_df.values, 
                     stratify=y_df.values, random_state=1)

model1 = LogisticRegression(solver='lbfgs')
model2 = KNeighborsClassifier(n_neighbors=3)
model3 = DecisionTreeClassifier()
ensemble = VotingClassifier(estimators=[('lr', model1), ('knn', model2), ('dt', model3)],
                                        voting='soft')


model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)
ensemble.fit(X_train, y_train)

print("LR 평가")
print("훈련 세트 정확도: {:.3f}".format(model1.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(model1.score(X_test, y_test)))

print("KNN 평가")
print("훈련 세트 정확도: {:.3f}".format(model2.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(model2.score(X_test, y_test)))

print("DT 평가")
print("훈련 세트 정확도: {:.3f}".format(model3.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(model3.score(X_test, y_test)))

print("VC 평가")
print("훈련 세트 정확도: {:.3f}".format(ensemble.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(ensemble.score(X_test, y_test)))



























