# -*- coding: utf-8 -*-

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.svm import SVC

# SVC 클래스의 하이퍼 파라메터 gamma
# auto 로 지정하는 경우 1 / n_features
# scale 로 지정하는 경우 1 / (n_features * X.std()) 
# - 스케일이 조정되지 않은 특성에서 좋은 결과를 반환
# - 특성 데이터를 전처리한 경우 scale과 auto는 동일함
# - 기본값은 auto
svc = SVC(gamma='auto')
svc.fit(X_train, y_train)

print("훈련 세트 정확도: {:.2f}".format(svc.score(X_train, y_train)))
print("테스트 세트 정확도: {:.2f}".format(svc.score(X_test, y_test)))














