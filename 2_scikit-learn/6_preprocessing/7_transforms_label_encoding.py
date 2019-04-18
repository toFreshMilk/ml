# -*- coding: utf-8 -*-

# 데이터전처리
# 데이터 분석을 위한 데이터 처리 과정
# - 전체 데이터 셋에서 데이터 분석에 사용될 열 선정
# - 특정 열에 존재하는 빈 값을 제거하거나
#   또는 특정 열에 존재하는 빈 값을 임의의 값으로 변경
# - 데이터의 스케일(값의 범위) 조정
# - 범주형 변수의 값 변경
#   (문자열 값의 수치 데이터화)
#   (원핫인코딩 처리)
# - 학습, 테스트 데이터 분할

import pandas as pd

iris = pd.read_csv("../../data/iris.csv")
print(iris.info())
print(iris.head())

# pandas 모듈을 사용하여 범주형 데이터를 수치 데이터로 변환 방법
# factorize 메소드
# 특정 열에 존재하는 모든 데이터의 중복을 제거한 후,
# 각 값에 대해서 정수값을 매핑하여 반환하는 메소드
# 반환값 -> 정수배열, 각 정수에 해당되는 실제 데이터
encoded, categories = iris.Species.factorize()
print(encoded)
print(categories)

# 범주데이터를 수치데이터로 변형하는 방법
# - LabelEncoder 클래스
# - 문자열이나 정수로된 라벨 값을  0  ~  K−1 까지의 정수로 변환 
# - 변환된 규칙은 classes_ 속성에서 확인할 수 있음
# - 예측 결과에 적용할 수 있도록 역변환을 위한 
#   inverse_transform 메서드를 지원
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
iris_le = encoder.fit_transform(iris.Species.values)

print(encoder.classes_)
print(encoder.inverse_transform([0]))

# 범주데이터를 수치데이터로 변형하는 방법
# - 원핫인코딩
# - 데이터의 대다수가 0으로 구성이 되며, 
#   특정 인덱스의 값만을 1로 갖도록 변환하는 인코딩 방법
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()

# print(iris.Species.values.reshape(-1,1).shape)
# 원핫 인코딩을 위한 배열의 재배치
# 원핫 인코딩은 반드시 2차원 배열로 데이터를
# 전달해야 합니다.
# iris 품종에 대한 일차원 배열을 2차원 배열로 변환하여
# 처리합니다.
iris_one_hot = encoder.fit_transform(iris.Species.values.reshape(-1,1))
print(type(iris_one_hot))
print(iris_one_hot)
print(iris_one_hot.toarray())
# 원핫인코딩으로 추출된 품종 정보를 추출
print(encoder.categories_)
# inverse_transform 메서드를 사용하여
# 원핫인코딩의 값을 실제 값으로 반환
# 주의사항 - 2차원 배열로 전달해야함
print(encoder.inverse_transform([[0,0,1]]))
























