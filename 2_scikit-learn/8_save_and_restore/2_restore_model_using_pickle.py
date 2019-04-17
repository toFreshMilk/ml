# -*- coding: utf-8 -*-

import pickle
with open("./save/save_model_using_pickle.bin", "rb") as f :
    clf = pickle.load(f)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(\
     cancer.data, cancer.target, random_state=0)

print("훈련 세트 정확도: {:.3f}".format(clf.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(clf.score(X_test, y_test)))





