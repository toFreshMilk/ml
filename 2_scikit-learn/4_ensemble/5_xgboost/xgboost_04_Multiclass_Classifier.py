# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import xgboost as xgb
            
wine = load_wine()

X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.3, random_state=42)

# 다항 분류를 위한 XGBClassifier 객체를 생성
# objective 하이퍼 파라메터의 값을 multi:softmax or multi:softprob 으로 설정
model = xgb.XGBClassifier(objective="multi:softprob", 
                          n_estimators=1000,
                          subsample=0.7,
                          random_state=42)
model.fit(X_train, y_train)

print("학습 결과 : ", model.score(X_train, y_train))
print("테스트 결과 : ", model.score(X_test, y_test))

y_pred = model.predict(X_train)
print("confusion_matrix - 학습 데이터")
print(confusion_matrix(y_train, y_pred))

y_pred = model.predict(X_test)
print("confusion_matrix - 테스트 데이터")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))











