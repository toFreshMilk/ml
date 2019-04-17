# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

X_df = pd.DataFrame(cancer.data)
y_df = pd.Series(cancer.target)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X_df.values, y_df.values, 
                     test_size=0.3, random_state=1)
    
# 선형모델을 사용하여 데이터를 분류할 수 있는 LogisticRegression 클래스
from sklearn.linear_model import LogisticRegression

# LogisticRegression 클래스는 기본 제약조건으로 L2 정규화를 지원
# 제약조건에 관련된 하이퍼 파라메터는 C 변수이며,
# 기본값은 1로 설정되어 있습니다.
# C의 값을 높일수록 제약의 강도가 낮아지며
# (학습 데이터를 더 많이 맞출 수 있음 - 과적합시킬 수 있음)
# C의 값은 낮출수록 제약의 강도가 높아집니다.
# (학습 데이터를 많이 맞추지 못하지만 테스트 데이터에 대한 일반화 성능이 높아짐)
lr001_model = LogisticRegression(solver='lbfgs', C=0.01,
                              max_iter=5000).fit(X_train, y_train)
lr_model = LogisticRegression(solver='lbfgs', C=1,
                              max_iter=5000).fit(X_train, y_train)
lr100_model = LogisticRegression(solver='lbfgs', C=100,
                              max_iter=5000).fit(X_train, y_train)

print("훈련 세트 점수(LR0.01): {:.3f}".format(lr001_model.score(X_train, y_train)))
print("훈련 세트 점수(LR1): {:.3f}".format(lr_model.score(X_train, y_train)))
print("훈련 세트 점수(LR100): {:.3f}".format(lr100_model.score(X_train, y_train)))

print("=" * 30)

print("테스트 세트 점수(LR0.01): {:.3f}".format(lr001_model.score(X_test, y_test)))
print("테스트 세트 점수(LR1): {:.3f}".format(lr_model.score(X_test, y_test)))
print("테스트 세트 점수(LR100): {:.3f}".format(lr100_model.score(X_test, y_test)))

from matplotlib import pyplot as plt

plt.plot(lr001_model.coef_.T, 'v', label="C=0.01")
plt.plot(lr_model.coef_.T, 'o', label="C=1")
plt.plot(lr100_model.coef_.T, '^', label="C=100")

plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)

xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.ylim(-5, 5)
plt.xlabel("ATTR")
plt.ylabel("COEF SIZE")
plt.legend()






























