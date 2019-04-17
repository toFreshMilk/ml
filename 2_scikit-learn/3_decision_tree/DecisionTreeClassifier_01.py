# -*- coding: utf-8 -*-

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
    train_test_split(cancer.data, cancer.target, 
                     stratify=cancer.target, test_size=0.3, 
                     random_state=1)

# 결정트리 알고리즘을 구현한 DecisionTreeClassifier 클래스
# 트리 구조를 사용하여 데이터를 분류 예측할 수 있는 클래스
from sklearn.tree import DecisionTreeClassifier

# 결정트리의 노드
# root node : 최상위 노드
# leaf node : 트리를 구성하고 있는 가장 하단의 노드들
# pure node : leaf node 중, 하나의 클래스, 값으로 분류된 노드
model = DecisionTreeClassifier(random_state=1).fit(X_train, y_train)

print("훈련 데이터 정확도 : ", model.score(X_train, y_train))
print("테스트 데이터 정확도 : ", model.score(X_test, y_test))


















