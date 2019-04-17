# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

X_df = pd.DataFrame(cancer.data)
y_df = pd.Series(cancer.target)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X_df.values, y_df.values, random_state=1)

# L1 제약 조건을 사용하는 경우 정규화 처리를 진행한 데이터에 대해서
# 성능이 감속할 수 있습니다.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.svm import LinearSVC
from matplotlib import pyplot as plt

# LinearSVC 클래스는 기본 제약조건으로 L2 정규화를 지원
# penalty 하이퍼 파라메터의 값을 l1으로 변경하면 
# 모델의 제약조건을 L1 정규화로 변경 할 수 있습니다.
# (dual 매개변수를 False 로 지정해야만 사용할 수 있음)

# C의 값을 높일수록 제약의 강도가 낮아지며
# (일부 특성 데이터의 가중치의 값만이 0으로 수렴)
# C의 값은 낮출수록 제약의 강도가 높아집니다.
# (대다수 특성 데이터의 가중치의 값이 0으로 수렴)

# 아래의 예는 제약조건을 L1으로 변경한 상태에서 모델을 테스트하는 예제
for C, marker in zip([0.01, 1, 100], ['o', '^', 'v']):
    svm_model_l1 = LinearSVC(C=C, penalty="l1", dual=False,
                            max_iter=100000).fit(X_train, y_train)
    print("C={:.3f} 인 l1 로지스틱 회귀의 훈련 정확도: {:.2f}".format(
          C, svm_model_l1.score(X_train, y_train)))
    print("C={:.3f} 인 l1 로지스틱 회귀의 테스트 정확도: {:.2f}".format(
          C, svm_model_l1.score(X_test, y_test)))
    
    plt.plot(svm_model_l1.coef_.T, marker, label="C={:.3f}".format(C))

plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.xlabel("ATTR")
plt.ylabel("COEF SIZE")
plt.ylim(-5, 5)
plt.legend(loc=3)







