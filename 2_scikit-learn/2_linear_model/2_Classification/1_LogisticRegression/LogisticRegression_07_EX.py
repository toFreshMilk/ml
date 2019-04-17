# -*- coding: utf-8 -*-

# LogisticRegression 클래스를 사용하여 load_iris 데이터를 분석하고
# 정확도 및 정밀도, 재현율을 확인하세요.

from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=1)

from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(C=1,
        solver='lbfgs', multi_class='multinomial', max_iter=5000).fit(X_train, y_train)

print("모델 평가(train) : ", lr_model.score(X_train, y_train))
print("모델 평가(test) : ", lr_model.score(X_test, y_test))

from sklearn.metrics import confusion_matrix, classification_report
pred_train = lr_model.predict(X_train)
pred_test = lr_model.predict(X_test)

print("훈련 데이터에 대한 confusion_matrix")
print(confusion_matrix(y_train, pred_train))
print(classification_report(y_train, pred_train))

print("테스트 데이터에 대한 confusion_matrix")
print(confusion_matrix(y_test, pred_test))
print(classification_report(y_test, pred_test))


















