# -*- coding: utf-8 -*-

from sklearn.datasets import load_boston

# 사이킷 런의 데이터를 직접 X와 y로 
# 전달받는 경우의 사용방법
X, y = load_boston(return_X_y=True)

print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, random_state=1)

from sklearn.linear_model import LinearRegression, Ridge, Lasso

# 분석 모델 클래스의 객체 생성
lr_model = LinearRegression().fit(X_train, y_train)
ridge_model = Ridge(alpha=1.7).fit(X_train, y_train)
lasso_model = Lasso(alpha=0.05).fit(X_train, y_train)

print("Lasso 모델의 가중치 : ", lasso_model.coef_)


# 분석 모델 객체의 평가(R2 스코어 확인 - 결정계수)
print("LR 평가 - train : ", lr_model.score(X_train, y_train))
print("Ridge 평가 - train : ", ridge_model.score(X_train, y_train))
print("Lasso 평가 - train : ", lasso_model.score(X_train, y_train))

print("=" * 30)

print("LR 평가 - test : ", lr_model.score(X_test, y_test))
print("Ridge 평가 - test : ", ridge_model.score(X_test, y_test))
print("Lasso 평가 - test : ", lasso_model.score(X_test, y_test))

from matplotlib import pyplot as plt

coef_range = list(range(1, len(ridge_model.coef_) + 1))

plt.plot(coef_range, lr_model.coef_, 'r^')
plt.plot(coef_range, ridge_model.coef_, 'bo')
plt.plot(coef_range, lasso_model.coef_, 'gv')

plt.hlines(0, 1, len(ridge_model.coef_) + 1, 
           colors='y', linestyles='dashed')

plt.show()



















