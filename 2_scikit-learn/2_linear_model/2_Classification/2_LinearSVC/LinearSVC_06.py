# -*- coding: utf-8 -*-

from sklearn.datasets import make_blobs

X, y = make_blobs(random_state=42)

#print(X[:10])
#print(y[:10])

from sklearn.svm import LinearSVC
model = LinearSVC(multi_class='ovr', max_iter=100000).fit(X, y)

# LinearSVC 클래스는 다항 분류를 지원하며, 
# 다항 분류를 위해서 일대다 방식을 적용
# - 각 클래스 별로 이진분류 모델을 생성하여
#   특성 데이터에 대한 각 이진분류 모델의 결과에서 가장 큰 값을 취함
# - 아래의 계수 배열과 같은 경우 3개의 클래스를 예측하기 위해서
#   2개의 특성을 사용하므로 3 X 2 열의 배열이 생성되며,
#   절편의 값 또한 각 클래스 별로 계산되므로 
#   3 의 크기를 갖는 일차원 배열이 반환
print("계수 배열의 크기: ", model.coef_.shape)
print("절편 배열의 크기: ", model.intercept_.shape)

print("모델 평가 : {:.3f}".format(model.score(X, y)))
print("모델의 예측 결과 : {}".format(model.predict(X)[:5]))
# 다항 분류의 경우 decision_function 메소드의 결과는
# 각 클래스에 해당할 확률 값이 반환되며,
# 그중 가장 큰 값으로 예측하게 됨
print("모델의 예측 결과(decision_function) : {}".format(model.decision_function(X)[:5]))

print("모델의 예측 결과 : {}".format(model.predict([[-6, 5]])))
print("모델의 예측 결과(decision_function) : {}".format(
        model.decision_function([[-6, 5]])))

















