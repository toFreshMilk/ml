# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import xgboost as xgb

pd.options.display.max_columns = 100
df = pd.read_csv("./data/food_dataset.csv")
            
# 정규화가 필요할 것으로 예측
# 값의 스케일이 특성간 차이가 심함
#print(df.describe())
# 결측데이터가 많이 발견됨
# 결측데이터를 제거해야함
#print(df.info())

X_df = df.iloc[:,:-1]
y_df = df.iloc[:,-1]

len(X_df.shape)
len(y_df.shape)

# 결측 데이터 제거
imputer = SimpleImputer()
X_df = pd.DataFrame(imputer.fit_transform(X_df.values))
print(X_df.info())

# 데이터 분할
X_train, X_test, y_train, y_test = \
train_test_split(X_df.values, y_df.values, 
                 test_size=0.3, random_state=21)

test_df = pd.DataFrame(y_test)
print(test_df[0].value_counts())

# 데이터 정규화
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

clf = xgb.XGBClassifier(objective="binary:logistic", 
                        n_estimators=500, 
                        learning_rate=0.2, subsample=0.7,
                        reg_lambda=2,
                        max_depth=10, random_state=42)
print(clf)
clf.fit(X_train, y_train)

print("학습 평가 점수 : ", clf.score(X_train, y_train))
print("테스트 평가 점수 : ", clf.score(X_test, y_test))

# 분류 평가
predictions = clf.predict(X_test)

print('Confusion Matrix:')
print(confusion_matrix(y_test, predictions))

print('Classification Report:')
print(classification_report(y_test, predictions))

from joblib import dump
dump(clf, "./save/food_dataset_clf.joblib")
dump(imputer, "./save/food_dataset_imputer.joblib")

















