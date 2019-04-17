# -*- coding: utf-8 -*-

# 랜덤포레스트(Random Forest)는 의사결정나무(Decision Tree)를 
# 개별 모형으로 사용하는 앙상블 방법

# 랜덤포레스트는 전체 입력 데이터 중 일부 열의 데이터만 선택하여 사용
# 하지만 노드 분리 시, 모든 독립 변수들을 비교하여 
# 최선의 독립 변수를 선택하는 것이 아니라 독립 변수 
# 차원을 랜덤하게 감소시킨 다음 그 중에서 독립 변수를 선택
# - 개별 모형들 사이의 상관관계를 감소시켜 
#   모형 성능의 변동을 최소화 할 수 있음

import numpy as np
from matplotlib import pyplot as plt 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.5, random_state=21)

model_1 = DecisionTreeClassifier(max_depth=10, 
                                random_state=0).fit(X_train, y_train)
model_2 = RandomForestClassifier(n_jobs=-1, max_depth=3, n_estimators=5000, 
                                random_state=0).fit(X_train, y_train)

print("model_1 정확도(학습 데이터) :", model_1.score(X_train, y_train))
print("model_2 정확도(학습 데이터) :", model_2.score(X_train, y_train))

print("model_1 정확도(테스트 데이터) :", model_1.score(X_test, y_test))
print("model_2 정확도(테스트 데이터) :", model_2.score(X_test, y_test))

predicted_1 = model_1.predict(X_test)

print('Confusion Matrix - 1:')
print(confusion_matrix(y_test, predicted_1))

print('Classification Report - 1 :')
print(classification_report(y_test, predicted_1))

predicted_2 = model_2.predict(X_test)

print('Confusion Matrix - 1:')
print(confusion_matrix(y_test, predicted_2))

print('Classification Report - 1 :')
print(classification_report(y_test, predicted_2))

# 각 독립 변수의 중요도(feature importance)를 계산
importances = model_2.feature_importances_

std = np.std([tree.feature_importances_ for tree in model_2.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

plt.title("feature_importances_")
plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()




























