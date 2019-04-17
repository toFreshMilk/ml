# -*- coding: utf-8 -*-

# data 디렉토리에 저장된 diabetes.csv 파일의 데이터를
# KNN 알고리즘을 사용하여 분석한 후, 정확도를 출력하세요

import pandas as pd

df = pd.read_csv("../../data/diabetes.csv", header=None)

#print(df.info())

X_df = df.iloc[:, :-1]
y_df = df.iloc[:, -1]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_df.values, 
                                                    y_df.values, 
                                                    test_size = 0.3, 
                                                    random_state=0)

print(f"train -> {len(y_train)}, test -> {len(y_test)}")

# KNN 알고리즘을 구현하고 있는 KNeighborsClassifier 클래스
from sklearn.neighbors import KNeighborsClassifier

K = 3
model = KNeighborsClassifier(n_neighbors=K)

model.fit(X_train, y_train)

predicted = model.predict(X_test)
print(predicted[:10])

# Score 메소드를 사용한 정확도 확인
print(f'Score : {model.score(X_test, y_test)} %')

# accuracy_score 함수를 사용한 정확도 확인
from sklearn.metrics import accuracy_score
print(f'Accuracy : {accuracy_score(y_test, predicted)} %')



















