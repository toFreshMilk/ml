# -*- coding: utf-8 -*-

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(\
     cancer.data, cancer.target, random_state=0)

# max_depth는 되도록 5이상은 주지않는 것이 일반적인 설정
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)

print("훈련 세트 정확도: {:.3f}".format(gbrt.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(gbrt.score(X_test, y_test)))

# learning_rate의 값은 내부적인 트리를 생성하는 것에 연관되며
# 낮은 learning_rate는 많은 하위 트리를 생성하게 됨
# 테스트를 통한 적절한 값을 찾는 것이 중요함
gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt.fit(X_train, y_train)

print("훈련 세트 정확도: {:.3f}".format(gbrt.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(gbrt.score(X_test, y_test)))

# subsample 파라메터는 딥러닝의 Drop-out 과 유사한 개념을 적용한 매개변수
# 전체 데이터를 학습하지 않고 일부분의 데이터를 사용하여
# 학습을 진행할 수 있도록 함
# (과적합을 방지하기 위한 제약 조건)
gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01, subsample=0.5)
gbrt.fit(X_train, y_train)

print("훈련 세트 정확도: {:.3f}".format(gbrt.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(gbrt.score(X_test, y_test)))

print(gbrt.feature_importances_)













