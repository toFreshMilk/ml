# -*- coding: utf-8 -*-

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(\
     cancer.data, cancer.target, random_state=0)

# max_depth는 되도록 5이상은 주지않는 것이 일반적인 설정
clf = GradientBoostingClassifier(random_state=0, max_depth=3, learning_rate=0.01, subsample=0.7)
clf.fit(X_train, y_train)

print("훈련 세트 정확도: {:.3f}".format(clf.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(clf.score(X_test, y_test)))

import os
from joblib import dump

try:
    if not(os.path.isdir("./save")):
        os.makedirs(os.path.join("./save"))
except OSError as e:
    if e.errno != errno.EEXIST:
        print("Failed to create directory!!!!!")
        raise

dump(clf, "./save/save_model_using_joblib.joblib")















