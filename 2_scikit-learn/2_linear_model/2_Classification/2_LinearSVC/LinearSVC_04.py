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

from sklearn.svm import LinearSVC

svm_model = LinearSVC(C=1, max_iter=100000).fit(X_train, y_train)

print("모델의 예측 결과 : ", svm_model.predict(X_test)[:5])

# LinearSVC 클래스는 확률 값을 반환하는 predict_proba 메소드가
# 제공되지 않음
#print("모델의 예측 결과(확률) : ", 
#      svm_model.predict_proba(X_test)[:5])

# decision_function 메소드를 사용하여 예측 결과의 과정을 
# 이해할 수 있음
# 0 을 기준으로 작다면 음성, 크다면 양성으로 예측함
pred = svm_model.decision_function(X_test[:5])
print('pred : ', pred)


















