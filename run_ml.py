from ravop.ml.linear_regression import LinearRegression
from sklearn.datasets import load_boston
import numpy as np

lr = LinearRegression(graph_id=1)

dataset = load_boston()

X = dataset.data
y = dataset.target[:, np.newaxis]

lr.train(X, y)


