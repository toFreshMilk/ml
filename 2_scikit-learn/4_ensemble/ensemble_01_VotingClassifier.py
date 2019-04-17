# -*- coding: utf-8 -*-

# 앙상블 방법론(ensemble methods)
# 모형 결합(model combining) : 다수개의 예측기를 결합하여 
# 하나의 예측 모델을 생성하는 방법
# 특정한 하나의 예측 방법이 아닌 복수의 예측 모형을 결합하여 
# 더 나은 성능의 예측을 하려는 시도로 나온 방법

# 단점
# 앙상블 방법은 일반적으로 머신러닝 모델의 계산량이 증가함

# 장점
# 단일 모형을 사용할 때 보다 성능 분산이 감소(과최적화를 방지)
# 개별 모형의 성능이 안좋을 경우에는 결합 모형의 성능이 더 향상

# 앙상블의 모형 결합을 위한 방법
# 취합(aggregation), 부스팅(boosting)

# 취합(aggregation) : 사용할 모형의 집합이 이미 결정되어 있는 경우
# - 다수결 (Majority Voting), 배깅 (Bagging), 랜덤포레스트 (Random Forests)
# 부스팅(boosting) : 사용할 모형을 점진적으로 늘려나가려는 경우
# - 에이다부스트 (AdaBoost), 그레디언트 부스트 (Gradient Boost)


# 다수결 (Majority Voting) 처리를 사용한 모델 생성 예제
# hard voting: 단순 투표. 개별 모형의 결과 기준
# soft voting: 가중치 투표. 개별 모형의 조건부 확률의 합 기준

# VotingClassifier 클래스
# - estimators: 예측기 목록, 리스트나 named parameter 형식을 지원
# - voting: 문자열 {hard, soft}
#  디폴트는 hard



import numpy as np
from matplotlib import pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

X = np.array([[0, -0.5], [-1.5, -1.5], [1, 0.5], [-3.5, -2.5], [0, 1], [1, 1.5], [-2, -0.5]])
y = np.array([1, 1, 1, 2, 2, 2, 2])
x_new = [0, -1.5]
plt.scatter(X[y == 1, 0], X[y == 1, 1], s=100, marker='o', c='r', label="CLASS 1")
plt.scatter(X[y == 2, 0], X[y == 2, 1], s=100, marker='x', c='b', label="CLASS 2")
plt.scatter(x_new[0], x_new[1], s=100, marker='^', c='g', label="TEST DATA")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Data for binary classification")
plt.legend()
plt.show()

model1 = LogisticRegression(solver='lbfgs')
model2 = KNeighborsClassifier(n_neighbors=3)
model3 = GaussianNB()
ensemble = VotingClassifier(estimators=[('lr', model1), ('knn', model2), ('gnb', model3)],
                                        voting='soft')

probas = [c.fit(X, y).predict_proba([x_new]) \
          for c in (model1, model2, model3, ensemble)]
class1_1 = [pr[0, 0] for pr in probas]
class2_1 = [pr[0, 1] for pr in probas]

ind = np.arange(4)
width = 0.35
p1 = plt.bar(ind, np.hstack(([class1_1[:-1], [0]])), 
             width, color='green')
p2 = plt.bar(ind + width, np.hstack(([class2_1[:-1], [0]])), 
             width, color='lightgreen')
p3 = plt.bar(ind, [0, 0, 0, class1_1[-1]], width, color='blue')
p4 = plt.bar(ind + width, [0, 0, 0, class2_1[-1]], width, 
             color='steelblue')

plt.xticks(ind + 0.5 * width, 
           ['LogisticRegression', 'KNN', 'Gaussian', 'Soft Voting'])
plt.ylim([0, 1.1])
plt.title('Result')
plt.legend([p1[0], p2[0]], ['CLASS 1', 'CLASS 2'], loc='upper left')
plt.show()






















