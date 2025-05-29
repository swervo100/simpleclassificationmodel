import sys
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import sklearn
import mglearn
from sklearn.datasets import load_iris
import sklearn.model_selection

iris_dataset = load_iris()
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] + "\n...")
print("Target names: {}".format(iris_dataset['target_names']))
print("Feature names: \n{}".format(iris_dataset['feature_names']))
print("Type of data: {}".format(type(iris_dataset['data'])))
print("Shape of data: {}".format(iris_dataset['data'].shape))
print("First five columns of data: \n{}".format(iris_dataset["data"][:5]))
print("Type of target: {}".format(type(iris_dataset['target'])))
print("Shape of target: {}".format(type(iris_dataset['target'].shape)))
print("Target:\n{}".format(iris_dataset['target']))
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)
print("X_train shape: {}".format(x_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(x_test.shape))
print("y_test shape: {}".format(y_test.shape))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
x_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(x_new.shape))
prediction = knn.predict(x_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(
    iris_dataset['target_names'][prediction]))
y_pred = knn.predict(x_test)
print("Test set predictions:\n {}".format(y_pred))
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
print("Test set score: {:.2f}".format(knn.score(x_test, y_test)))
