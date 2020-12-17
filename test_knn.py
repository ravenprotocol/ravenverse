
from sklearn.datasets import load_iris
from ravop.ml.knn import KNN
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np


dataset = load_iris()

X = dataset.data

y = dataset.target

mu = np.mean(X, 0)
sigma = np.std(X, 0)
X = (X - mu ) / sigma

X_train, X_test, y_train, y_test = train_test_split(\
                X, y, test_size=0.3, random_state=45)

our_classifier = KNN(X_train, y_train, n_neighbours=3, n_classes=3, weights="distance")
sklearn_classifier = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)

our_accuracy = our_classifier.score(X_test, y_test)
sklearn_accuracy = sklearn_classifier.score(X_test, y_test)

print(" \n\n DISPLAY ACUURACY COMPARISON CHART \n\n", pd.DataFrame([[our_accuracy, sklearn_accuracy]],
             ['Accuracy'],
             ['Our Implementation', 'Sklearn\'s Implementation']))

print("DONE -> Let's Move to Ravop")