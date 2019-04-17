# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import xgboost as xgb

from joblib import load
# 예측기와 전처리 객체를 로딩
clf = load("./save/food_dataset_clf.joblib")
imputer = load("./save/food_dataset_imputer.joblib")

pd.options.display.max_columns = 100
df = pd.read_csv("./data/food_dataset.csv")

X_df = df.iloc[:,:-1]
y_df = df.iloc[:,-1]

# 결측 데이터 제거
X_df = pd.DataFrame(imputer.transform(X_df.values))
print(X_df.info())

# 데이터 분할
X_train, X_test, y_train, y_test = \
train_test_split(X_df.values, y_df.values, 
                 test_size=0.3, random_state=21)

# 분류 평가

predictions = clf.predict(X_test)

print('Confusion Matrix:')
print(confusion_matrix(y_test, predictions))

print('Classification Report:')
print(classification_report(y_test, predictions))



















