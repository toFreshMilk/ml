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

# 선형모델을 사용하여 데이터를 분류할 수 있는 LogisticRegression 클래스
from sklearn.linear_model import LogisticRegression

# 선형 Support Vector Machine 알고리즘을 구현하고 있는
# LinearSVC 클래스(Linear Support Vector Classification)
from sklearn.svm import LinearSVC

lr_model = LogisticRegression(solver='lbfgs', max_iter=5000).fit(X_train, y_train)

# Linear Support Vector Machine 알고리즘으로 구현된 분류 클래스
# Support Vector Machine을 구현하고 있는 SVC에 비해서
# 선형 계산에 특화되어 있어 선형 데이터를 분류하는 경우 더 효율적
# (LinearSVC 클래스의 학습이후, the number of iterations 메세지가
# 출력되는 경우 학습이 완료되지 않은 상태이므로 max_iter 매개변수를
# 조정하여 성능을 높일 수 있습니다.)
svm_model = LinearSVC(max_iter=50000).fit(X_train, y_train)

print("훈련 세트 점수(LR): {:.3f}".format(lr_model.score(X_train, y_train)))
print("훈련 세트 점수(SVM): {:.3f}".format(svm_model.score(X_train, y_train)))

print("=" * 30)

print("테스트 세트 점수(LR): {:.3f}".format(lr_model.score(X_test, y_test)))
print("테스트 세트 점수(SVM): {:.3f}".format(svm_model.score(X_test, y_test)))
































