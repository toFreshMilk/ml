# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import xgboost as xgb

df = pd.read_csv("../../../data/diabetes.csv")

X_df = df.iloc[:,:-1]
y_df = df.iloc[:,-1]

# 데이터 분할
X_train, X_test, y_train, y_test = \
train_test_split(X_df.values, y_df.values, 
                 test_size=0.3, stratify=y_df.values, random_state=21)

model = xgb.XGBClassifier(objective="binary:logistic", random_state=42,
                          n_estimators=10000, max_depth=5, 
                          reg_lambda=1000, reg_alpha=10)

model.fit(X_train, y_train)

print("학습 평가 점수 : ", model.score(X_train, y_train))
print("테스트 평가 점수 : ", model.score(X_test, y_test))

# 분류 평가
predictions = model.predict(X_test)

print('Confusion Matrix:')
print(confusion_matrix(y_test, predictions))

print('Classification Report:')
print(classification_report(y_test, predictions))


















