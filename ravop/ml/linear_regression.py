import logging
import logging.handlers

import numpy as np

from common import db
from ravop import globals as g
from ravop.core import Graph, Tensor, Scalar, Op
from ravop.socket_client import SocketClient


class LinearRegression(object):
    def __init__(self, **kwargs):
        self.__setup_logger()

        # If graph id is provided, fetch the old graph
        graph_id = g.graph_id
        if graph_id is not None:
            self.graph = Graph(id=graph_id)
        else:
            self.graph = Graph()
            g.graph_id = self.graph.id

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
        self.graph.clean()

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

        # 2. Calculate cost
        cost = self.__compute_cost(y, y_pred, no_samples)

        # 3. Gradient descent - Update weight values
        for i in range(iter):
            y_pred = X.matmul(weights, name="y_pred{}".format(i))
            c = X.trans().matmul(y_pred)
            d = learning_rate.div(no_samples)
            weights = weights.sub(c.elemul(d), name="weights{}".format(i))
            cost = self.__compute_cost(y, y_pred, no_samples, name="cost{}".format(i))

        return cost, weights

    def __compute_cost(self, y, y_pred, no_samples, name="cost"):
        # Calculate cost
        a = y_pred.sub(y)
        b = a.elemul(a).matsum()
        cost = Scalar(1).div(Scalar(2).elemul(no_samples)).elemul(b, name=name)
        return cost

    def print_ops(self):
        # Print ops
        ops = db.get_graph_ops(self.graph.id)
        ops = [Op(id=op.id) for op in ops]

        for op in ops:
            print(op)

    def __str__(self):
        return "LinearRegression:Graph Id:{}\n".format(self.graph.id)
