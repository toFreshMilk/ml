# -*- coding: utf-8 -*-

# 부스팅 (boosting)
# - 성능이 약한 학습기(weak learner) 다수 개를 연결하여 
#   강한 학습기(strong learner)를 만드는 앙상블 학습 방법
# - 부스팅 방법은 앞서 학습된 약한 모델의 약점을 보완해나가면서 
#   성능이 개선된 더 나은 모델로 학습시키는 것

# 부스팅 방법론을 구현하고 있는 모델
# 1. 아다부스트(AdaBoost, Adaptive Boosting)
# - 아다부스트(AdaBoost)는 과소적합(underfitted)된 
#  학습 데이터 샘플의 가중치를 높이면서 새로 학습된 모델이 
#  학습하기 어려운 데이터에 더 잘 적합되도록 하는 방식
# - 아다부스트의 학습 단계
# a. 전제 학습 데이터셋을 이용해 모델을 만든 후, 
#    잘못 예측(분류)된 샘플의 가중치를 상대적으로 증가시킴
# b. 다음 모델을 학습 시킬 때 업데이트된 가중치가 반영된 데이터를 사용하여 
#    모델을 학습 시킨다. 
# c. a와 b의 과정을 반복

# Scikit-learn의 AdaBoostClassifier 클래스를 사용


# 2. 그래디언트 부스팅(Gradient Boosting)
# - 아다부스트와 마찬가지로 직전에 학습된 모델의 오차를 보완하는 방향으로 
# 모델(분류기, 학습기)을 추가해주는 방법
# - 그래디언트 부스팅은 아다부스트와 달리 학습단계 마다 데이터 샘플의 
#   가중치를 업데이트 하는 것이 아니라 
#   직전 단계 모델의 잔여 오차에 대해 새로운 모델을 학습시키는 방법

# Scikit-learn의 GradientBoostingClassifier 클래스를 사용

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(\
     cancer.data, cancer.target, random_state=0)

ada_model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                               random_state=0).fit(X_train, y_train)
gb_model = GradientBoostingClassifier(max_depth=3, 
                                      random_state=0).fit(X_train, y_train)

print("훈련 세트 정확도(ADA): {:.3f}".format(ada_model.score(X_train, y_train)))
print("훈련 세트 정확도(GB): {:.3f}".format(gb_model.score(X_train, y_train)))

print("테스트 세트 정확도(ADA): {:.3f}".format(ada_model.score(X_test, y_test)))
print("테스트 세트 정확도(GB): {:.3f}".format(gb_model.score(X_test, y_test)))






















