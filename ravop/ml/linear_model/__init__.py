import logging
import logging.handlers

from common import db
from ravop.core import Tensor, Graph
import ravop.core as R
from ravop import globals as g
from ravop.ml import metrics


class OrdinaryLeastSquares(Graph):
    def __init__(self, id=None, **kwargs):
        super().__init__(id=id, **kwargs)

        self.__setup_logger()

        self._coefficients = None
        self._intercepts = []

    def __setup_logger(self):
        # Set up a specific logger with our desired output level
        self.logger = logging.getLogger(OrdinaryLeastSquares.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

        # Add the log message handler to the logger
        handler = logging.handlers.RotatingFileHandler(g.ravop_log_file)

        self.logger.addHandler(handler)

    def fit(self, X, y):
        # Convert input values to RavOp tensors
        X = Tensor(X, name="X")
        y = Tensor(y, name="y")

        self._coefficients = R.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y, name="coefficients")

    def predict(self, X):
        """
        Predict
        """
        X = Tensor(X, name="X")
        return X.dot(self._coefficients)

    @property
    def coefficients(self):
        if self._coefficients is None:
            self._coefficients = db.get_ops_by_name(op_name="coefficients", graph_id=self.id)[0]
        print(self._coefficients.id)

        if self._coefficients.status == "computed":
            return self._coefficients.output
        else:
            raise Exception("Need to train the model first")

    @property
    def intercepts(self):
        return self._intercepts

    def score(self, X, y, name="r2"):
        g.graph_id = None
        y_pred = self.predict(X)
        y_true = y

        if name == "r2":
            return metrics.r2_score(y_true, y_pred)
        else:
            return None

    def __str__(self):
        return "OrdinaryLeastSquares:Graph Id:{}\n".format(self.id)

