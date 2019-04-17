# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("./data/winequality-red.csv", sep=";")
X_df = df.iloc[:,:-1]
y_df = df.iloc[:,-1]    

X_train, X_test, y_train, y_test = \
train_test_split(X_df.values, y_df.values, 
                 test_size=0.5, random_state=21)

model_1 = DecisionTreeClassifier(criterion='entropy',
                                random_state=0).fit(X_train, y_train)

model_2 = BaggingClassifier(DecisionTreeClassifier(criterion='entropy', 
                           max_depth=3), n_estimators=1000, 
                           random_state=0).fit(X_train, y_train)

print("model_1 정확도(학습 데이터) :", model_1.score(X_train, y_train))
print("model_2 정확도(학습 데이터) :", model_2.score(X_train, y_train))

print("model_1 정확도(테스트 데이터) :", model_1.score(X_test, y_test))
print("model_2 정확도(테스트 데이터) :", model_2.score(X_test, y_test))

predicted_1 = model_1.predict(X_test)

print('Confusion Matrix - 1:')
print(confusion_matrix(y_test, predicted_1))

print('Classification Report - 1 :')
print(classification_report(y_test, predicted_1))

predicted_2 = model_2.predict(X_test)

print('Confusion Matrix - 2:')
print(confusion_matrix(y_test, predicted_2))

print('Classification Report - 2 :')
print(classification_report(y_test, predicted_2))

















