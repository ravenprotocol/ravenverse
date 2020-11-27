import logging
import logging.handlers

import numpy as np

from ravop import globals as g
from ravop.core import Graph, Tensor, Scalar
from ravop.ml import metrics
import ravop.core as R


class LinearRegression(Graph):
    def __init__(self, id=None, **kwargs):
        super().__init__(id=id, **kwargs)

        self.__setup_logger()

        # Define hyperparameters
        self.learning_rate = kwargs.get("learning_rate", None)
        if self.learning_rate is None:
            self.learning_rate = 0.03

    def __setup_logger(self):
        # Set up a specific logger with our desired output level
        self.logger = logging.getLogger(LinearRegression.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

        # Add the log message handler to the logger
        handler = logging.handlers.RotatingFileHandler(g.ravop_log_file)

        self.logger.addHandler(handler)

    def train(self, X, y, iter=10):
        self.clean()

        # Convert input values to RavOp tensors
        X = Tensor(X, name="X")
        y = Tensor(y, name="y")

        # Initialize params
        learning_rate = Scalar(self.learning_rate)
        size = X.shape[1]
        no_samples = Scalar(X.shape[0])
        weights = Tensor(np.random.uniform(0, 1, size).reshape((size, 1)), name="weights")

        # 1. Predict
        y_pred = X.matmul(weights, name="y_pred")

        # 2. Compute cost
        cost = self.__compute_cost(y, y_pred, no_samples)

        # 3. Gradient descent - Update weight values
        for i in range(iter):
            y_pred = X.matmul(weights, name="y_pred{}".format(i))
            c = X.transpose().matmul(y_pred)
            d = learning_rate.div(no_samples)
            weights = weights.sub(c.elemul(d), name="weights{}".format(i))
            cost = self.__compute_cost(y, y_pred, no_samples, name="cost{}".format(i))

        return cost, weights

    def predict(self, x):
        """Predict values"""
        weights = self.weights

        # Local predict
        return np.matmul(x, weights)

    def __compute_cost(self, y, y_pred, no_samples, name="cost"):
        """Cost function"""
        a = y_pred.sub(y)
        b = a.elemul(a).matsum()
        cost = Scalar(1).div(Scalar(2).multiply(no_samples)).elemul(b, name=name)
        return cost

    @property
    def weights(self):
        """Retrieve weights"""
        ops = self.get_ops_by_name(op_name="weight", graph_id=self.id)
        if len(ops) == 0:
            raise Exception("You need to train your model first")

        # Get weights
        weight_op = ops[-1]
        if weight_op.status == "pending" or weight_op.status == "computing":
            raise Exception("Please wait. Your model is getting trained")

        weight = weight_op.output

        return weight

    def score(self, X, y, name="r2"):
        y_pred = self.predict(X)
        y_true = y

        if name == "r2":
            return metrics.r2_score(y_true, y_pred)
        else:
            return None

    def __str__(self):
        return "LinearRegression:Graph Id:{}\n".format(self.id)
