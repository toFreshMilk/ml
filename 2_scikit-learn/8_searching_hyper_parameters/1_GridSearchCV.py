# -*- coding: utf-8 -*-

# 최적의 하이퍼 파라메터를 검색하기 위한 예제
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

iris = load_iris()

X_train, X_test, y_train, y_test = \
    train_test_split(iris.data, iris.target, random_state=0)
    
best_score = 0

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        # 각 매개변수의 조합에 대해서 SVC 모델 객체를 생성하여 훈련
        svm = SVC(gamma=gamma, C=C).fit(X_train, y_train)
        # 테스트 세트를 사용하여 SVC 모델을 평가
        score = svm.score(X_test, y_test)
        # 평가 점수가 높은 경우 해당 SVC 모델과 매개변수를 저장
        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}
            
print("최고 점수: {:.2f}".format(best_score))
print("최적 파라미터: {}".format(best_parameters))