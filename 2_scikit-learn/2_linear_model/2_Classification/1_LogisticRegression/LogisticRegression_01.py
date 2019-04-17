# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

X_df = pd.DataFrame(cancer.data)
y_df = pd.Series(cancer.target)

pd.options.display.max_columns = 100

# 상황에 따라 정규화의 필요성이 있는 특성 데이터
print(X_df.describe())
# 양성과 음성데이터의 편향이 높음
print(y_df.value_counts())
print(y_df.value_counts() / len(y_df))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X_df.values, y_df.values, 
                     test_size=0.3, random_state=1)
    
# 최근접 이웃 알고리즘을 사용하여 분류할 수 있는 클래스의 로딩
from sklearn.neighbors import KNeighborsClassifier
# 선형모델을 사용하여 데이터를 분류할 수 있는 LogisticRegression 클래스
from sklearn.linear_model import LogisticRegression

K = 7
knn_model = KNeighborsClassifier(n_neighbors=K).fit(X_train, y_train)

# LogisticRegression 클래스의 하이퍼 파라메터 solver
# 학습을 위해서 사용되는 알고리즘을 선택할 수 있는 파라메터
# 기본 값은 liblinear
# - 작은 데이터 셋에 잘 동작하는 알고리즘으로 L1, L2 정규화를 지원
# sag, saga
# - 대용량의 데이터를 빠르게 학습할 수 있는 알고리즘
# 다중 클래스의 분류 모델은 newton-cg, sag, saga 과 lbfgs 를 사용해야함
# newton-cg, lbfgs, sag 알고리즘은 L2 정규화만 지원
# liblinear, saga 알고리즘은 L1 정규화도 지원함

lr_model = LogisticRegression(solver='lbfgs', max_iter=5000).fit(X_train, y_train)

print("훈련 세트 점수(KNN): {:.3f}".format(knn_model.score(X_train, y_train)))
print("훈련 세트 점수(LR): {:.3f}".format(lr_model.score(X_train, y_train)))

print("=" * 30)

print("테스트 세트 점수(KNN): {:.3f}".format(knn_model.score(X_test, y_test)))
print("테스트 세트 점수(LR): {:.3f}".format(lr_model.score(X_test, y_test)))
































