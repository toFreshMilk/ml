# -*- coding: utf-8 -*-

import xgboost as xgb
print(xgb.__version__)

# XGBoost(Extreme Gradient Boosting)
# https://xgboost.readthedocs.io/en/latest/index.html
# 캐글에서 많은 빈도로 사용되는 데이터 분석을 위한 부스팅 기반의 라이브러리
# 캐글 : 데이터 과학자들이 통계적 문제를 놓고 경쟁하는 온라인 플랫폼

# XGBoost의 장점
# -병렬 처리를 사용하여 학습과 분류가 Gradient Boosting에 비해 빠름
# -평가 함수 및 다양한 커스텀 최적화 옵션을 제공
# -알고리즘 내에서 자동으로 제공하는 가지치기(과적합 방지 기능)로 인하여
#  과적합(Overfitting)이 잘 일어나지 않음
# -기존의 다른 알고리즘과 연계 활용성 뛰어남
#  (xgboost 분류기 결론부 아래에 다른 알고리즘을 붙여서 앙상블 학습이 가능)

# 기본 동작 방식
# 기본적으로 부스팅(Boosting, or Additive Training) 기술을 사용
# 부스팅 : 약한 분류기를 세트로 묶어서 정확도를 예측하는 기법

# xgboost는 욕심쟁이(Greedy algorithm)를 사용하여 내부의 약한 분류기들을
# 발견하고, 분산처리를 사용하여 빠른(Extreme) 속도로 적합한 비중 파라메터를 
# 찾는 알고리즘

# xgboost의 파라메터
# 1. 일반 파라메터
# - booster: 부스터 구조를 결정하기 위한 파라메터
# (gbtree(기본값), gblinear, dart)
# - nthread: 쓰레드의 사용 개수
# (디폴트는 가능한 최대수)
# - num_feature: feature 차원의 숫자를 정해야 하는 경우 사용
# (디폴트는 가능한 최대수)

# 2. 부스팅 파라메터
# - learning rate: 학습률, 트리에 가지(노드)가 많을수록 과적합(overfitting)
#   하기 쉽기 때문에 부스팅 스텝마다 weight를 주어 부스팅 과정에서 
#   과적합이 일어나지 않도록 제어
# - gamma: 정보 획득(Information Gain), 값이 클수록 트리 깊이가 줄어들어 
#   보수적인 모델이 된다. 디폴트 값은 0(반드시 조정해야함)
# - subsample: 각 트리마다의 관측 데이터 샘플링 비율, 
#   값을 작게 하는 경우 over-fitting을 방지하지만 너무 작은 값은 under-fitting.
#   일반적으로 0.5-1(디폴트 1)
# - max_depth: 각 트리의 maximum depth. 
#   숫자를 키울수록 모델의 복잡도가 높아짐. 과적합에 빠지기 쉬움.
#   디폴트 값은 6(이 경우 리프 노드의 개수는 최대 2^6 = 64개)
# - reg_lambda(L2 reg-form): L2 Regularization Form weights. 
#   값이 커질 수록 보수적인 모델이 된다
# - reg_alpha(L1 reg-form): L1 Regularization Form weights. 
#   값이 커질 수록 보수적인 모델이 된다 

# 3. 학습 과정 파라메터
# - objective: 목적 함수
#   <reg:linear, binary:logistic, multi:softmax, multi:softprob>
# - eval_metric: 모델의 평가 함수
#   <rmse(root mean square error), logloss(log-likelihood), 
#    map(mean average precision) 등이 제공됨>

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import xgboost as xgb
            
diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.3, random_state=42)

#model = xgb.XGBRegressor(objective="reg:linear", 
#                         max_depth=7,
#                         n_estimators=500,
#                         subsample=0.8,
#                         reg_lambda=1.5,
#                         random_state=42)
model = xgb.XGBRegressor(objective="reg:linear",                          
                         random_state=42)
print(model)

model.fit(X_train, y_train)

y_pred = model.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
print("학습 결과 : ", np.sqrt(mse))

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("테스트 결과 : ", np.sqrt(mse))








