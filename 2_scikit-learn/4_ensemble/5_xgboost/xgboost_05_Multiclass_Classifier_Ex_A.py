# -*- coding: utf-8 -*-

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import xgboost as xgb

df = pd.read_csv("../../../data/winequality-red.csv", sep=";")
#df = pd.read_csv("../../../data/winequality-white.csv", sep=";")

X_df = df.iloc[:,:-1]
y_df = df.iloc[:,-1]    

X_train, X_test, y_train, y_test = \
train_test_split(X_df.values, y_df.values, 
                 test_size=0.3, stratify=y_df.values, random_state=21)

model = xgb.XGBClassifier(objective="multi:softprob", random_state=42)

#model = xgb.XGBClassifier(objective="multi:softprob", random_state=42,
#                          n_estimators=1000)

#model = xgb.XGBClassifier(objective="multi:softprob", random_state=42,
#                          n_estimators=1000, max_depth=2)

#model = xgb.XGBClassifier(objective="multi:softprob", random_state=42,
#                          n_estimators=10000, max_depth=5, subsample=0.7,
#                          reg_lambda=100)

model.fit(X_train, y_train)

print("학습 데이터 정확도 : ", model.score(X_train, y_train))
print("테스트 데이터 정확도 : ", model.score(X_test, y_test))

# 분류 평가
#predictions = model.predict(X_test)
#
#print('Confusion Matrix:')
#print(confusion_matrix(y_test, predictions))
#
#print('Classification Report:')
#print(classification_report(y_test, predictions))









