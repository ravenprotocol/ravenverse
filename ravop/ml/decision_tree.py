from ravop.core import Tensor, Scalar
import numpy as np


class BaseAlgorithm(object):
    def __init__(self):
        self.params = None
        self.accuracy = None

    def train(self, X, y=None):
        pass

    def predict(self, X):
        pass

    def set_params(self, **kwargs):
        self.params = kwargs

    def get_params(self):
        return self.params

    def get_accuracy(self):
        return self.accuracy


class DecisionTreeClassifier(BaseAlgorithm):
    def __init__(self):
        super().__init__()

    def train(self, X, y=None):
        # Convert input values to RavOp tensors
        X = Tensor(X, name="X")
        y = Tensor(y, name="y")

        # 2. Train
        # 3. Accuracy

        size = X.shape[1]
        no_samples = Scalar(X.shape[0])
        weights = Tensor(np.random.uniform(0, 1, size).reshape((size, 1)), name="weights")

        y_pred = X.matmul(weights)
        # Decision tree

    def predict(self, X):
        pass

    def calculate_accuracy(self, y_pred, y_true, metric=None):
        pass

    @property
    def feature_importance(self):
        return None


class DecisionTreeRegressor(BaseAlgorithm):
    def __init__(self):
        super().__init__()

    def train(self, X, y=None):
        # 1. Split
        # 2. Train
        # 3. Accuracy
        pass

    def predict(self, X):
        pass
