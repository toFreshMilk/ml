# -*- coding: utf-8 -*-

# winequality-red.csv 데이터를 BaggingClassifier 클래스를 이용하여 
# 분석한 후, 결과를 확인하세요.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report, confusion_matrix

fname = '../../data/winequality-red.csv'
df = pd.read_csv(fname, sep=';')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, 
                     stratify=y.values, test_size=0.3, 
                     random_state=21)

model = BaggingClassifier(DecisionTreeClassifier(max_depth=5), 
                        n_estimators=50000,
                        random_state=0).fit(X_train, y_train)

print("Bagging 정확도(학습 데이터) :", model.score(X_train, y_train))
print("Bagging 정확도(테스트 데이터) :", model.score(X_test, y_test))

predicted = model.predict(X_test)

print('Confusion Matrix:')
print(confusion_matrix(y_test, predicted))





