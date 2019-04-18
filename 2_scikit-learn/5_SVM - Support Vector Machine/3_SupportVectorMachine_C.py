# -*- coding: utf-8 -*-

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

import pandas as pd
X = pd.DataFrame(cancer.data)

pd.options.display.max_columns = 100
print(X.describe())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

# SVM 알고리즘을 기반으로 하는 예측기들은
# 특성 데이터 스케일에 많은 영향을 받습니다.
# 만약 특성 데이터의 스케일이 서로 다른 영역에 위치한 경우
# 공간의 분할이 어려워 지므로 학습이 올바르게 진행되지 않습니다.
# SVM 알고리즘 기반의 예측기를 사용하는 경우
# 반드시 데이터의 전처리를 수행해야만 올바르게 학습이 이뤄집니다.
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.svm import SVC

# SVC 클래스의 하이퍼 파라메터 C
svc = SVC(gamma='auto', C=1000)
svc.fit(X_train, y_train)

print("훈련 세트 정확도: {:.2f}".format(svc.score(X_train, y_train)))
print("테스트 세트 정확도: {:.2f}".format(svc.score(X_test, y_test)))














