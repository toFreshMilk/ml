# -*- coding: utf-8 -*-

# Data 디렉토리에 저장된 diabetes.csv 파일의 데이터를
# 분석하여 정확도 및 정밀도, 재현율을 출력하세요.
# (LinearSVC 클래스를 활용하되, C의 값과 penalty를
# 제어하여 결과를 확인)

import pandas as pd

fname = '../../../../data/diabetes.csv'
df = pd.read_csv(fname, header=None)

X_df = df.iloc[:, :-1]
y_df = df.iloc[:,  -1]

print(y_df.value_counts() / len(y_df))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X_df.values, y_df.values, stratify=y_df.values,
                     test_size=0.3, random_state=1)

from sklearn.svm import LinearSVC

#model = LinearSVC(C=1, max_iter=20000, penalty='l1', dual=False).fit(X_train, y_train)
model = LinearSVC(C=10, max_iter=50000).fit(X_train, y_train)

print("훈련 세트 점수(SVC) : ", model.score(X_train, y_train))
print("=" * 30)
print("테스트 세트 점수(SVC) : ", model.score(X_test, y_test))

from sklearn.metrics import confusion_matrix, classification_report
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

print("훈련 데이터에 대한 confusion_matrix")
print(confusion_matrix(y_train, pred_train))
print(classification_report(y_train, pred_train))

print("테스트 데이터에 대한 confusion_matrix")
print(confusion_matrix(y_test, pred_test))
print(classification_report(y_test, pred_test))


