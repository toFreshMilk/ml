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

# 결정트리 알고리즘을 사용하는 경우 주의사항
# 결정트리 알고리즘은 학습 데이터에 과적합(overfittng)되는 경향을 보입니다.
# 결정트리 알고리즘을 사용하는 경우 반드시 과적합을 방지하기 위한
# 하이퍼 파라메터를 적용해야 합니다.

# 트리구조에서 과적합(overfittng)을 방지하는 방법
# 1. tree의 생성을 사전에 차단하는 방법
# - 특정 조건을 지정하여 완벽하게 학습하지 못하도록 방지
# 2. tree의 생성이 완료된 후, 특정 leaf 노드들을 삭제하거나
#   병합하는 방법

# 사이킷 런에서는 1번의 방법만을 지원(사전 차단 방법을 지원)
model = DecisionTreeClassifier(max_depth=3, random_state=1).fit(X_train, y_train)

print("훈련 데이터 정확도 : ", model.score(X_train, y_train))
print("테스트 데이터 정확도 : ", model.score(X_test, y_test))

# 결정트리 모델의 학습 결과를 시각화하기 위한 설정
# 1. 현재 운영체제에 맞는 graphviz 설치 
# - http://www.graphviz.org/
# 2. PATH 등록
# - graphviz 설치된 디렉토리의 bin 경로
# - 사용자 변수, 시스템 변수 모두의 Path 변수에 경로 추가
# 3. graphviz 파이썬 모듈 설치
# - pip install graphviz

from sklearn.tree import export_graphviz

export_graphviz(model, out_file='cancer_tree.dot', 
                class_names=['악성', '양성'], 
                feature_names=cancer.feature_names, filled=True)

import graphviz
from IPython.display import display

with open('cancer_tree.dot', encoding='utf-8') as f:
    dot_graph = f.read()
    
display(graphviz.Source(dot_graph))

print(len(cancer.data))
print(len(X_train))



















