# -*- coding: utf-8 -*-

import pandas as pd

fname = '../../data/winequality-red.csv'
df = pd.read_csv(fname, sep=';')

X_df = df.iloc[:,:-1]
y_df = df.iloc[:, -1]

print("type(X_df) : ", type(X_df))
print("type(y_df) : ", type(y_df))

print(y_df.value_counts())
print(y_df.value_counts() / len(y_df))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
train_test_split(X_df.values, y_df.values, random_state=1)

print("len(X_train) : ", len(X_train))
print("len(X_test) : ", len(X_test))

print(y_train[:10])
print(y_test[:10])












