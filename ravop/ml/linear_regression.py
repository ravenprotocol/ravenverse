import logging
import logging
import logging.handlers

import numpy as np

from common.db_manager import NodeTypes, OpTypes, Operators
from ravop.constants import RAVOP_LOG_FILE
from ravop.graph_manager import GraphManager
from ravop.socket_client import SocketClient


class LinearRegression(object):
    def __init__(self, graph_id=None):
        self.graph_id = graph_id

        # Create database client
        self.graph_manager = GraphManager(graph_id=graph_id)

        # Create a socket client instance
        self.socket_client = SocketClient().connect()

        # Define Hyper-parameters
        self.iterations = 1
        self.learning_rate = 0.03
        self.epsilon = 1e-5

        self.weights = None
        self.ops = dict()

        self.logger = None
        self.__setup_logger()

    def clean(self):
        self.socket_client.disconnect()

    def __setup_logger(self):
        # Set up a specific logger with our desired output level
        self.logger = logging.getLogger(LinearRegression.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

        # Add the log message handler to the logger
        handler = logging.handlers.RotatingFileHandler(RAVOP_LOG_FILE)

        self.logger.addHandler(handler)

    def __init_weights(self, size):
        print("Initializing weights")
        # Instantiate weight values
        # weights = np.zeroes((X.shape[1], 1))
        self.weights = np.random.uniform(0, 1, size).reshape((size, 1))
        print("Weights initialized")

    def train(self, X, y):
        # Step 0
        # Clean
        self.graph_manager.clear_graph()

        # Step 1
        # Initialize weights
        self.__init_weights(X.shape[1])

        # Step 2
        # Create data ops
        data_list = [{"name": "X", "data": X, "type": "ndarray"},
                     {"name": "y", "data": y, "type": "ndarray"},
                     {"name": "weights", "data": self.weights, "type": "ndarray"},
                     {"name": "one", "data": 1, "type": "integer"},
                     {"name": "epsilon", "data": self.epsilon, "type": "float"},
                     {"name": "no_samples", "data": X.shape[0], "type": "integer"},
                     {"name": "learning_rate", "data": self.learning_rate, "type": "float"}
                     ]

        data_ops = self.graph_manager.create_data_ops(graph_id=self.graph_id,
                                                      data_list=data_list)

        self.ops.update(data_ops)

        # Step 3
        # Create predict ops
        y_pred_op = self.__create_predict_ops(X_op=self.ops["X"], weights_op=self.ops["weights"])
        y_true_op = self.ops['y']

        print("OPs created")

        self.clean()

    def predict(self, X):
        pass

    def __create_predict_ops(self, X_op, weights_op):
        # 1. Create an operation to multiply X and params
        y_pred_op = self.graph_manager.create_op(self.graph_id, name="y_pred", node_type=NodeTypes.MIDDLE.value,
                                                 inputs=[X_op.id, weights_op.id],
                                                 outputs=None, op_type=OpTypes.BINARY.value,
                                                 operator=Operators.MATRIX_MULTIPLICATION.value)

        self.ops["y_pred"] = y_pred_op
        return y_pred_op
