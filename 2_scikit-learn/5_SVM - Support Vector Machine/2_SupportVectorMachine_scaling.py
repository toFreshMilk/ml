# -*- coding: utf-8 -*-

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

from sklearn.svm import SVC

# SVC 클래스의 하이퍼 파라메터 C
# 모델의 학습에 제약을 설정하기 위한 파라메터
# C의 값이 낮을 수록 강한 제약이 설정됨
# (모델이 과정합된 경우 일반화 성능을 높이기 위해서 사용)
# C의 값이 높을 수록 약한 제약이 설정됨
# (모델이 과소적합된 경우 학습 성능을 높이기 위해서 사용)
svc = SVC(gamma='auto', C=0.1, random_state=0)
svc.fit(X_train, y_train)

print("훈련 세트 정확도: {:.2f}".format(svc.score(X_train, y_train)))
print("테스트 세트 정확도: {:.2f}".format(svc.score(X_test, y_test)))



























