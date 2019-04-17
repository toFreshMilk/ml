# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("./data/winequality-red.csv", sep=";")
X_df = df.iloc[:,:-1]
y_df = df.iloc[:,-1]    

y_df.value_counts()

X_train, X_test, y_train, y_test = \
train_test_split(X_df.values, y_df.values, 
                 test_size=0.3, random_state=21)

model1 = LogisticRegression(solver='lbfgs', multi_class='multinomial')
model2 = KNeighborsClassifier(n_neighbors=3)
model3 = GaussianNB()
ensemble = VotingClassifier(estimators=[('lr', model1), 
                                        ('knn', model2), 
                                        ('gnb', model3)], voting='soft')

predictions = [c.fit(X_train, y_train).predict(X_test)\
               for c in (model1, model2, model3, ensemble)]

train_score = []
test_score = []

train_score.append(model1.score(X_train, y_train))
test_score.append(model1.score(X_test, y_test))
print('Accuracy (LogisticRegression) - train :', model1.score(X_train, y_train))
print('Accuracy (LogisticRegression) - test :', model1.score(X_test, y_test))
print('Confusion Matrix(LogisticRegression) :')
print(confusion_matrix(y_test, predictions[0]))
print('Classification Report(LogisticRegression):')
print(classification_report(y_test, predictions[0]))

train_score.append(model2.score(X_train, y_train))
test_score.append(model2.score(X_test, y_test))
print('Accuracy (KNeighborsClassifier) - train :', model2.score(X_train, y_train))
print('Accuracy (KNeighborsClassifier) - test :', model2.score(X_test, y_test))
print('Confusion Matrix(KNeighborsClassifier) :')
print(confusion_matrix(y_test, predictions[1]))
print('Classification Report(KNeighborsClassifier):')
print(classification_report(y_test, predictions[1]))
    
train_score.append(model3.score(X_train, y_train))
test_score.append(model3.score(X_test, y_test))
print('Accuracy (GaussianNB) - train :', model3.score(X_train, y_train))
print('Accuracy (GaussianNB) - test :', model3.score(X_test, y_test))
print('Confusion Matrix(GaussianNB) :')
print(confusion_matrix(y_test, predictions[2]))
print('Classification Report(GaussianNB):')
print(classification_report(y_test, predictions[2]))

train_score.append(ensemble.score(X_train, y_train))
test_score.append(ensemble.score(X_test, y_test))
print('Accuracy (VotingClassifier) - train :', ensemble.score(X_train, y_train))
print('Accuracy (VotingClassifier) - test :', ensemble.score(X_test, y_test))
print('Confusion Matrix(VotingClassifier) :')
print(confusion_matrix(y_test, predictions[3]))
print('Classification Report(VotingClassifier):')
print(classification_report(y_test, predictions[3]))

from matplotlib import pyplot as plt 
ind = np.arange(4)
plt.bar(ind, train_score)
plt.show()
plt.bar(ind, test_score, color="r")
plt.show()

















