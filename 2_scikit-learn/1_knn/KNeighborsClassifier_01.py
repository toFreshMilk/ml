# -*- coding: utf-8 -*-

import pandas as pd
# 유방암의 음성/양성 데이터
# 이진 분류 데이터
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

X_df = pd.DataFrame(cancer.data)
y_df = pd.DataFrame(cancer.target)

print(X_df.describe())
print(y_df[0].value_counts())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
train_test_split(X_df.values, y_df[0].values, random_state=1)

# 최근접 이웃 알고리즘을 구현하고 있는 예측기를 로딩
# 예측기 : 사이킷런에서 제공하는 데이터 학습을 위한 클래스
from sklearn.neighbors import KNeighborsClassifier

# KNeighborsClassifier : K 최근접 이웃 알고리즘을 구현하고
# 있는 클래스
# 학습이 굉장히 빠른 예측기(단순 저장만을 하기때문에...)
# 예측에 사용될 데이터를 저장하고 있는 데이터와 거리를 비교하여
# 분류를 수행하는 예측기

# 예측기 객체의 생성
# 사이킷런의 모든 예측기 클래스들은 각 알고리즘에 따라
# 서로 다른 하이퍼 파라메터를 가지고 있음
# 하이퍼 파라메터 : 사용자가 직접 지정하여 설정하는 값
# 하이퍼 파라메터에 따라서 성능이 변화됨
model = KNeighborsClassifier(n_neighbors=7)

# 예측기의 데이터 학습
# 사이킷 런의 모든 예측기 클래스들은
# fit 메소드를 사용하여 데이터를 학습
# - fit(X, y) 의 형태로 사용
# 주의사항
# - X(입력데이터)는 반드시 2차원으로 입력
# - y(라벨데이터)는 반드시 1차원으로 입력
model.fit(X_train, y_train)

# 예측기의 성능 평가
# 사이킷런의 모든 예측기 클래스는 score 메소드를 제공
# score 메소드는 예측기의 분류에 따라서 서로 다른 값을 반환
# 분류 모델(Classifier) : 정확도를 반환
# 회귀 모델() : R2 score를 반환
print("학습 데이터 평가 : ", model.score(X_train, y_train))
print("테스트 데이터 평가 : ", model.score(X_test, y_test))

# 예측기를 사용한 예측 결과 반환
# 사이킷 런의 모든 예측기 클래스는 predict 메소드를 제공
# predict 메소드는 2차원 배열의 데이터를 입력받아
# 예측 결과를 1차원 배열로 반환하는 메소드입니다.
predicted = model.predict(X_test[:3])
print(predicted)
print(y_test[:3])

# 예측기를 사용한 예측 결과 반환
# 사이킷 런의 분류를 위한 예측기 클래스들은
# predicted_proba 메소드를 제공
# predicted_proba 메소드는 분류할 각 클래스에 대한
# 입력데이터의 확률값을 반환(가장 큰 값이 예측값)
predicted_proba = model.predict_proba(X_test[:3])
print(predicted_proba)













