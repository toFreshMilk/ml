# -*- coding: utf-8 -*-

# winequality-red.csv 데이터의 교차 검증 점수를 확인하세요
# 머신러닝 모델은 분류 모델을 사용합니다.

import pandas as pd
df = pd.read_csv("../../data/winequality-red.csv", sep=";")
X_df = df.iloc[:,:-1]
y_df = df.iloc[:,-1]    
print(df.info())
print(df.describe())
print(X_df)

from sklearn.model_selection import train_test_split

#y의 비율로 분할 stratify=y_df.values
X_train, X_test, y_train, y_test = \
    train_test_split(X_df.values, y_df.values, 
                     stratify=y_df.values, random_state=0)
    

from sklearn.preprocessing import MinMaxScaler
#테스트데이터에서는 fit이 아닌 transform만 사용
scaler = MinMaxScaler().fit(X_train)
X_train_scaler = scaler.transform(X_train)
X_test_scaler = scaler.transform(X_test)

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold

model = SVC(gamma='scale')

#그냥 셔플이 아닌 비율을 맞춘 셔플
kfold = KFold(n_splits=5, shuffle=True, random_state=1)
scores = cross_val_score(model, X_train_scaler, y_train, cv=kfold)
print("교차 검증 점수(KFold 3) : {}".format(scores))




