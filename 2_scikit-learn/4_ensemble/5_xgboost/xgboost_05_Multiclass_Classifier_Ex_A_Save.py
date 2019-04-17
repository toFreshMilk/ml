# -*- coding: utf-8 -*-

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import xgboost as xgb

df = pd.read_csv("./data/winequality-red.csv", sep=";")

X_df = df.iloc[:,:-1]
y_df = df.iloc[:,-1]    

X_train, X_test, y_train, y_test = \
train_test_split(X_df.values, y_df.values, 
                 test_size=0.3, random_state=21)

clf = xgb.XGBClassifier(objective="multi:softprob", random_state=42,
                        n_estimators=3000, max_depth=10, subsample=0.8,
                        reg_lambda=1.5, learning_rate=0.25)

#clf = xgb.XGBClassifier(objective="multi:softprob", n_estimators=1000, 
#                        learning_rate=0.15, max_depth=10, subsample=0.8, 
#                        reg_lambda=1.5, reg_alpha=1.5, random_state=42)

clf.fit(X_train, y_train)

print("학습 데이터 정확도 : ", clf.score(X_train, y_train))
print("테스트 데이터 정확도 : ", clf.score(X_test, y_test))

# 분류 평가
predictions = clf.predict(X_test)

print('Confusion Matrix:')
print(confusion_matrix(y_test, predictions))

print('Classification Report:')
print(classification_report(y_test, predictions))

from joblib import dump
dump(clf, "./save/winequality-red_clf.joblib")






