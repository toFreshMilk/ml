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

# 사이킷런의 변환기 클래스들은 상속을 기반으로
# 동작하지 않습니다. (덕 타이핑 기반으로 동작하는 방식을 사용)
# - 덕 타이핑 : 상속이나 인터페이스 구현이 아니라 
#   객체의 속성이나 메서드가 객체의 유형을 결정하는 방식
# 변환기로 동작하기 위한 클래스들은 
# fit, transform, fit_transform 메소드를 정의해야 합니다.
# (fit_transform 메소드를 정의하지 않은 경우
# fit 메소드의 호출 후 transform 메소드가 자동호출)

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector (BaseEstimator, TransformerMixin) :
    def __init__(self, attr_names) :
        # 데이터프레임에서 추출할 컬럼명의 리스트
        self.attr_names = attr_names
        
    def fit(self, X, y=None) :
        return self
    
    def transform(self, X) :
        # 데이터프레임에서 생성자로 전달받은
        # 컬럼명의 데이터들을 numpy 배열로 반환
        return X[self.attr_names].values

scores = pd.read_csv("../../data/score.csv")

print(scores.info())
print(scores.columns)

selector = DataFrameSelector(scores.columns[1:])
selector.fit(scores)
print(selector.transform(scores))





























