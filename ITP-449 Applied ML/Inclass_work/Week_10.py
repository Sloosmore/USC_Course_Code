"""
    Stan Loosmore
    ITP-449
    Week 10 - ML: 
"""

import pandas as pd 
import numpy as np 
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


def main():
    petal = pd.read_csv('Inclass_work/csv_folders/iris.csv')
    print(petal.info())
    X = petal.drop(columns='Species')
    y = petal['Species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, random_state=123)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    plot_tree(model, filled=True, feature_names=X.columns, class_names=y.unique())
    plt.savefig('Inclass_work/figs/Decistion_Tree.png')


main()