# -*- coding: utf-8 -*-

# SVM(Support Vector Machine)은 러시아 과학자 Vladimir Vapnik이
# 1970년대 후반에 제안한 알고리즘
# 분류(classification)문제에서 우수한 일반화(generalization) 성능이 입증되어 
# 머신러닝 알고리즘에서 범용적으로 활용
# SVM 기반의 머신러닝 알고리즘들은 일반화 성능이 다른 분류 모델과 비교할 때
# 더 좋거나 대등한 것으로 알려져 있다.

# SVM 기반의 머신러닝 알고리즘들은 선형 또는 비선형 분류 뿐만아니라 
# 회귀, 이상치 탐색에도 사용할 수 있음
# 복잡한 분류 문제에도 성능이 우수하며, 중간 크기의 데이터셋에 적합함

# SVM 기반의 머신러닝 알고리즘들은 데이터를 분류하기 위한
# 최대 여백(margin - 마진)의 값을 찾는 방법을 제공함
# 여백(마진)을 최대화하는 결정 초평면(decision hyperplane)을 찾는 것이 
# SVM 기반 모델들의 목표

# 사이킷 런에서는 LinearSVC, SVC, SVR 클래스 등이 제공됨

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

from sklearn.svm import SVC, LinearSVC

# SVC 클래스의 하이퍼 파라메터 gamma
# SVM 알고리즘을 사용하여 각 영역을 구분할 때
# 마진의 크기를 결정하는 하이퍼 파라메터
# gamma 의 값이 작을수록 큰 마진의 값을 찾음
# (학습데이터에 과적합된 경우 일반화 성능을 높이기 위해서 사용)
# gamma 의 값이 커질수록 작은 마진의 값을 찾음
# (학습데이터에 대한 성능을 높이기 위해서 사용)
# auto 로 지정하는 경우 1 / n_features
# scale 로 지정하는 경우 1 / (n_features * X.std()) 
# - 스케일이 조정되지 않은 특성에서 좋은 결과를 반환
# - 특성 데이터를 전처리한 경우 scale과 auto는 동일함
# - 기본값은 auto

#svc = LinearSVC(C=0.1, max_iter=10000, random_state=0)

svc = SVC(gamma=0.001, random_state=0)

svc.fit(X_train, y_train)

print("훈련 세트 정확도: {:.2f}".format(svc.score(X_train, y_train)))
print("테스트 세트 정확도: {:.2f}".format(svc.score(X_test, y_test)))














