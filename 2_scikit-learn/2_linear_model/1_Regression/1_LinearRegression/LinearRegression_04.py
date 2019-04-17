# -*- coding: utf-8 -*-

import numpy as np

X = np.array([6, 8, 10, 14, 18]).reshape(-1,1)
y = np.array([7, 9, 13, 17.5, 18])

from sklearn.linear_model import LinearRegression

# LinearRegression 클래스는 학습의 결과인
# 가중치와 절편을 coef_와 intercept_ 변수에 저장

model = LinearRegression()

# 가중치와 절편의 값은 fit 메소드의 실행 이후에
# 접근할 수 있는 멤버입니다.
#print("기울기(가중치) : {0}, 절편 : {1}".format(
#        model.coef_, model.intercept_))

model.fit(X, y)

# fit 메소드의 실행 이후에는 학습데이터에 대한
# 기울기와 절편의 값을 확인할 수 있음
print("기울기(가중치) : {0}, 절편 : {1}".format(
        model.coef_, model.intercept_))

pred_1 = model.predict(X)
pred_2 = X.reshape(-1) * model.coef_ + model.intercept_

print("predict 메소드를 사용하여 반환받은 결과")
print(pred_1)
print("가중치(기울기)와 절편을 사용하여 반환받은 결과")
print(pred_2)





















