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

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(solver='lbfgs', C=1,
                              max_iter=5000).fit(X_train, y_train)

print("모델의 예측 결과 : ", lr_model.predict(X_test)[:5])
print("모델의 예측 결과(확률) : ", 
      lr_model.predict_proba(X_test)[:5])




















