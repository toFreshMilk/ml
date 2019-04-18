# -*- coding: utf-8 -*-

# 최적의 하이퍼 파라메터를 검색하기 위한 예제
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC

iris = load_iris()

X_train, X_test, y_train, y_test = \
    train_test_split(iris.data, iris.target, random_state=0)
    
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

print("매개변수 그리드:\n{}".format(param_grid))

#교차검증 지원하면 CV붙음
from sklearn.model_selection import GridSearchCV

kfold = KFold(n_splits=5, shuffle=True)
# iid 샘플수의 가중치조절해줌
grid_search = GridSearchCV(
        SVC(), param_grid, cv=kfold, return_train_score=True, iid=True)

grid_search.fit(X_train, y_train)

print("테스트 세트 점수: {:.2f}".format(grid_search.score(X_test, y_test)))

print("최적 매개변수: {}".format(grid_search.best_params_))

print("최고 교차 검증 점수: {:.2f}".format(grid_search.best_score_))

print("최고 성능 모델:\n{}".format(grid_search.best_estimator_))









