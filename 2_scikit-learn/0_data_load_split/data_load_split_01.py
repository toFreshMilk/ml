# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.datasets import load_wine

# 사이킷 런에서 제공하는 훈련데이터 로딩
wine = load_wine()

print(wine.keys())
# 분류 데이터인 경우 예측한 결과의 
# 이름을 확인할 수 있도록 target_names
# 키가 제공됨
print(wine.target_names)

# 입력(특성) 데이터
X_df = pd.DataFrame(wine.data)
# 라벨(정답) 데이터
y_df = pd.DataFrame(wine.target)

# 입력데이터의 개수 및 타입, 
# 결측데이터 확인
print(X_df.info())

pd.options.display.max_columns = 100
# 입력데이터의 모든 수치값을 확인
print(X_df.describe())

# 라벨데이터의 빈도 확인
print(y_df[0].value_counts())
print(y_df[0].value_counts() / len(y_df))


# 학습 전 단계
# 1. 데이터의 분할
# - 학습 데이터, 테스트 데이터, 검증 데이터
# - 학습 데이터 : 머신러닝 모델이 학습할 데이터
# - 테스트 데이터 : 학습이 종료된 머신러닝 모델이
#  정답을 예측하기 위한 데이터
#  (머신러닝 모델의 일반화 정도를 판단하는 기준이 됨)
# - 검증 데이터 : 딥러닝 모델과 같이 단계별로 학습을
#   진행하는 경우 일정 단계에서 검증을 위한 목적으로
#   사용되는 데이터
#   학습데이터의 정확도와 검증데이터 정확도의 추이를
#   비교하여 학습 도중 과적합 여부를 판단
# - 학습(70%), 테스트(20%), 검증(10%)

# 사이킷 런의 데이터 분할을 위해서 제공되는
# train_test_split 함수
from sklearn.model_selection import train_test_split

# train_test_split 함수의 사용법
# train_test_split(X, y, 추가적인 파라메터정보...)
# random_state : 난수 발생의 seed 값을 의미
# - 동일한 데이터와 동일한 random_state 정보가 대입되면
#   항상 동일한 데이터 셋이 반환되도록 보장할 수 있음
#   (다수번의 학습 시 비교를 수월하게 진행할 수 있음)
# test_size : 테스트 데이터 셋의 비율(실수의 값 사용)
# - 0.3이 입력되는 경우, 학습데이터 70%. 
#   테스트데이터 30%가 반환
# - test_size 를 지정하지 않은 경우
#   학습데이터 75%. 테스트데이터 25%가 반환

# train_test_split 함수의 반환 값
# X_train(학습할 입력데이터), X_test(테스트할 입력데이터), 
# y_train(학습할 라벨데이터), y_test(테스트할 라벨데이터) 
# = train_test_split(...)

# 실제 사용 예
# pandas 데이터 프레임에서 numpy 배열을 반환받는 방법
# values 속성을 사용하여 numpy 배열을 반환받을 수 있음
# 주의 사항
# 사이킷런의 모든 학습을 위한 클래스들은
# 입력데이터는 2차원 배열, 라벨데이터는 1차원으로 가정
X_train, X_test, y_train, y_test = \
    train_test_split(X_df.values, y_df[0].values, 
                     random_state=1)

# 데이터의 분할 비율을 확인
print(X_train.shape)
print(X_test.shape)
# 라벨 데이터의 분할 정보 확인
print(y_train[:10])
print(y_test[:10])


















