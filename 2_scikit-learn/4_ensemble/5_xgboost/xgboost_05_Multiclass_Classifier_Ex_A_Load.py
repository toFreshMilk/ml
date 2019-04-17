# -*- coding: utf-8 -*-

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import xgboost as xgb

from joblib import load
clf = load("./save/winequality-red_clf.joblib")

df = pd.read_csv("./data/winequality-red.csv", sep=";")
X_df = df.iloc[:,:-1]
y_df = df.iloc[:,-1]    

X_train, X_test, y_train, y_test = \
train_test_split(X_df.values, y_df.values, 
                 test_size=0.3, random_state=21)

print("학습 데이터 정확도 : ", clf.score(X_train, y_train))
print("테스트 데이터 정확도 : ", clf.score(X_test, y_test))

# 분류 평가
predictions = clf.predict(X_test)

print('Confusion Matrix:')
print(confusion_matrix(y_test, predictions))

print('Classification Report:')
print(classification_report(y_test, predictions))








