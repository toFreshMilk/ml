# -*- coding: utf-8 -*-

# winequality-red.csv 데이터를 RandomForestClassifier 클래스를 이용하여 
# 분석한 후, 결과를 확인하세요.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

fname = '../../../data/winequality-red.csv'
df = pd.read_csv(fname, sep=';')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, 
                     stratify=y.values, test_size=0.3, 
                     random_state=21)

#model = RandomForestClassifier(random_state=0, n_jobs=-1).fit(X_train, y_train)
#model = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1).fit(X_train, y_train)
#model = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1).fit(X_train, y_train)
#model = RandomForestClassifier(n_estimators=1000, max_features=2, random_state=0, n_jobs=-1).fit(X_train, y_train)
#model = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=0, n_jobs=-1).fit(X_train, y_train)

# min_samples_split : 각 노드의 분기 시, 최소 샘플의 개수
#model = RandomForestClassifier(n_estimators=10000, min_samples_split=5, random_state=0, n_jobs=-1).fit(X_train, y_train)

model = RandomForestClassifier(n_estimators=1000000, max_depth=10, random_state=0, n_jobs=-1).fit(X_train, y_train)

print("RandomForest 정확도(학습 데이터) :", model.score(X_train, y_train))
print("RandomForest 정확도(테스트 데이터) :", model.score(X_test, y_test))

predicted = model.predict(X_test)

print('Confusion Matrix:')
print(confusion_matrix(y_test, predicted))












