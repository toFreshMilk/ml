# -*- coding: utf-8 -*-

# LinearSVC 클래스를 사용하여 load_wine 데이터를 분석하고
# 정확도 및 정밀도, 재현율을 확인하세요.

import pandas as pd
from sklearn.datasets import load_wine

wine = load_wine()

X_df = pd.DataFrame(wine.data)
y_df = pd.Series(wine.target)

print(X_df.describe())
print(y_df.value_counts() / len(y_df))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X_df.values, y_df.values, 
                     stratify=y_df.values, test_size=0.3, random_state=1)

# L2 정규화를 사용하여 데이터를 분석하는 경우
# 특성데이터의 스케일을 압축할 수 있는 데이터 정규화를 진행하는 것으
# 성능 향상에 도움이 됩니다.
# L1 정규화를 사용하는 경우 데이터 스케일을 압축할 필요가 없음
#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler().fit(X_train)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)

from sklearn.svm import LinearSVC
#model = LinearSVC(C=100, max_iter=1000000).fit(X_train, y_train)
model = LinearSVC(C=100, max_iter=1000000, penalty="l1", dual=False).fit(X_train, y_train)

print("모델 평가(train) : ", model.score(X_train, y_train))
print("모델 평가(test) : ", model.score(X_test, y_test))

from sklearn.metrics import confusion_matrix, classification_report
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

print("훈련 데이터에 대한 confusion_matrix")
print(confusion_matrix(y_train, pred_train))
print(classification_report(y_train, pred_train))

print("테스트 데이터에 대한 confusion_matrix")
print(confusion_matrix(y_test, pred_test))
print(classification_report(y_test, pred_test))


















