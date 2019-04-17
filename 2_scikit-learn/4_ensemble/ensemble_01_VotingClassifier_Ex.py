# -*- coding: utf-8 -*-

# winequality-red.csv 데이터를 LogisticRegression, KNeighborsClassifier, 
# GaussianNB 를 조합한 VotingClassifier로 분석한 후, 결과를 확인하세요.


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






