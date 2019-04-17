# -*- coding: utf-8 -*-

# DecisionTreeClassifier 클래스를 사용하여 load_wind 데이터를 분석하고
# 정확도 및 정밀도, 재현율을 확인하세요.
# (DecisionTree의 그래프, 특성 중요도를 시각화하여 확인하세요)                 

import pandas as pd
from sklearn.datasets import load_wine

wine = load_wine()

X_df = pd.DataFrame(wine.data)
y_df = pd.Series(wine.target)

print(y_df.value_counts() / len(y_df))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
    train_test_split(X_df.values, y_df.values, 
                     stratify=y_df.values, random_state=1)
    
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train)

print("훈련 세트 정확도: {:.3f}".format(model.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(model.score(X_test, y_test)))

# print("특성 중요도:\n{}".format(model.feature_importances_))

import numpy as np
from matplotlib import pyplot as plt

def plot_feature_importances_cancer(model):
    n_features = wine.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), wine.feature_names)
    plt.xlabel("feature_importances")
    plt.ylabel("feature")
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(model)

wine.keys()
from sklearn.tree import export_graphviz

export_graphviz(model, out_file='wine_tree.dot', 
                class_names=wine.target_names, 
                feature_names=wine.feature_names, filled=True)

import graphviz
from IPython.display import display

with open('wine_tree.dot', encoding='utf-8') as f:
    dot_graph = f.read()
    
display(graphviz.Source(dot_graph))


















