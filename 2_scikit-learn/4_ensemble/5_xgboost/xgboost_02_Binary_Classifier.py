# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import xgboost as xgb
            
cancer = load_breast_cancer()

X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.3, random_state=42)

# 이진 분류를 위한 XGBClassifier 객체 생성
# objective 하이퍼 파라메터의 값을 binary:logistic 으로 설정
clf = xgb.XGBClassifier(objective="binary:logistic", 
                        random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_train)
print("confusion_matrix - 학습 데이터")
print(confusion_matrix(y_train, y_pred))

y_pred = clf.predict(X_test)
print("confusion_matrix - 테스트 데이터")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))













