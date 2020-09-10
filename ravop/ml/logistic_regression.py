import logging
import logging.handlers

import numpy as np

from ravop import globals as g
from ravop.core import Graph, Tensor, Scalar


class LogisticRegression(Graph):
    def __init__(self, id=None, **kwargs):
        super().__init__(id=id, **kwargs)

        self.__setup_logger()

        # Define hyperparameters
        self._learning_rate = kwargs.get("learning_rate", None)
        if self._learning_rate is None:
            self._learning_rate = 0.03

    def __setup_logger(self):
        # Set up a specific logger with our desired output level
        self.logger = logging.getLogger(LogisticRegression.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

        # Add the log message handler to the logger
        handler = logging.handlers.RotatingFileHandler(g.ravop_log_file)

        self.logger.addHandler(handler)

    def train(self, X, y, iter=10):
        # Remove old ops and start from scratch
        self.clean()

        # Convert input values to RavOp tensors
        X = Tensor(X, name="X")
        y = Tensor(y, name="y")

        # Initialize params
        learning_rate = Scalar(self._learning_rate)
        size = X.shape[1]
        no_samples = Scalar(X.shape[0])
        weights = Tensor(np.random.uniform(0, 1, size).reshape((size, 1)), name="weights")

        # 1. Predict - Calculate y_pred
        y_pred = self.sigmoid(X.matmul(weights), name="y_pred")

        # 2. Compute cost
        cost = self.__compute_cost(y, y_pred, no_samples)

        for i in range(iter):
            y_pred = self.sigmoid(X.matmul(weights), name="y_pred{}".format(i))
            weights = weights.sub(learning_rate.div(no_samples).elemul(X.trans().matmul(y_pred.sub(y))),
                                  name="weights{}".format(i))
            cost = self.__compute_cost(y=y, y_pred=y_pred, no_samples=no_samples, name="cost{}".format(i))

        return cost, weights

    def predict(self, x):
        """Predict values"""
        weights = self.weights

        # Local predict
        return 1 / (1 + np.exp(-np.matmul(x, weights)))

    def __compute_cost(self, y, y_pred, no_samples, name="cost"):
        """Cost function"""
        epsilon = Scalar(1e-5)
        one = Scalar(1)

        c1 = y.neg().trans().matmul(y_pred.add(epsilon).natlog())
        c2 = one.sub(y).trans().matmul(one.sub(y_pred).add(epsilon).natlog())
        cost = one.div(no_samples).elemul(c1.sub(c2), name=name)
        return cost

    def sigmoid(self, x, name="sigmoid"):
        """Sigmoid activation function"""
        # 1/(1+e^-x)
        one = Scalar(1)
        return one.div(x.neg().exp().add(one), name=name)

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

    def __str__(self):
        return "LogisticRegression:Graph Id:{}\n".format(self.id)
