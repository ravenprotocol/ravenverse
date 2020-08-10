import json
import logging
import logging.handlers

import numpy as np

from common.db_manager import Op, NodeTypes, Data, OpTypes, Operators, OpStatus, DBManager, Graph
from ravop.graph_manager import GraphManager
from ..constants import RAVOP_LOG_FILE


class LogisticRegression(object):
    def __init__(self, graph_id=None):
        # Create database client
        self.db = DBManager()
        self.graph_manager = GraphManager(graph_id=graph_id)

        # Hyper-parameters
        self.iterations = 1
        self.learning_rate = 0.03
        self.epsilon = 1e-5

        self.weights = None
        self.ops = dict()

        self.logger = None
        self.__setup_logger()

    def __setup_logger(self):
        # Set up a specific logger with our desired output level
        self.logger = logging.getLogger(LogisticRegression.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

        # Add the log message handler to the logger
        handler = logging.handlers.RotatingFileHandler(RAVOP_LOG_FILE)

        self.logger.addHandler(handler)

    def __init_weights(self, size):
        # Instantiate weight values
        # weights = np.zeroes((X.shape[1], 1))
        self.weights = np.random.uniform(0, 1, size).reshape((size, 1))

    def train(self, X, y):
        # Initialize weights
        self.__init_weights(X.shape[1])

        data_list = [{"name": "X", "data": X, "type": "ndarray"},
                     {"name": "y", "data": y, "type": "ndarray"},
                     {"name": "weights", "data": self.weights, "type": "ndarray"},
                     {"name": "one", "data": 1, "type": "integer"},
                     {"name": "epsilon", "data": self.epsilon, "type": "float"},
                     {"name": "no_samples", "data": X.shape[0], "type": "integer"},
                     {"name": "learning_rate", "data": self.learning_rate, "type": "float"}
                     ]

        # data_dict = {"X": X, "y": y, "weights": self.weights, "one": 1, "epsilon": self.epsilon,
        #              "no_samples": X.shape[0]}

        data_ops = self.__create_data_ops(data_list=data_list)

        self.ops.update(data_ops)

        # Predict
        y_pred_op = self.__create_predict_ops(X_op=self.ops["X"], weights_op=self.ops["weights"])
        y_true_op = self.ops['y']

        # Calculate cost ops
        self.__create_cost_ops(y_true_op=y_true_op, y_pred_op=y_pred_op)

        # Create ops to calculate stocastic gradient descent
        final_cost_op, updated_weights_op = self.__create_sgd_ops(X_op=self.ops["X"], y_op=self.ops['y'], weights_op=self.ops["weights"])


    def __create_predict_ops(self, X_op, weights_op):

        # 1. Create an operation to multiply X and params
        op1 = self.__create_op(name="op1", node_type=NodeTypes.MIDDLE.value,
                               inputs=[X_op.id, weights_op.id],
                               outputs=None, op_type=OpTypes.BINARY.value,
                               operator=Operators.MATRIX_MULTIPLICATION.value)

        self.ops["op1"] = op1

        # Calulcate sigmoid
        # Numpy - 1 / (1 + np.exp(-x))

        # Operations
        # 2. Negate x
        op2 = self.__create_op(name="op2", node_type=NodeTypes.MIDDLE.value, inputs=[op1.id], outputs=None,
                               op_type=OpTypes.UNARY.value, operator=Operators.NEGATION.value)

        self.ops["op2"] = op2

        # 3. Exponential of the output of op2
        op3 = self.__create_op(name="op3", node_type=NodeTypes.MIDDLE.value, inputs=[op2.id], outputs=None,
                               op_type=OpTypes.UNARY.value, operator=Operators.EXPONENTIAL.value)

        self.ops["op3"] = op3

        # 4. Add numeric 1 to the output of step 2
        op4 = self.__create_op(name="op4", node_type=NodeTypes.MIDDLE.value, inputs=[op3.id, self.ops["one"].id],
                               outputs=None,
                               op_type=OpTypes.BINARY.value, operator=Operators.ADDITION.value)

        self.ops["op4"] = op4

        # 5. Divide numeric 1 by op4
        op5 = self.__create_op(name="y_pred", node_type=NodeTypes.MIDDLE.value, inputs=[self.ops["one"].id, op4.id],
                               outputs=None,
                               op_type=OpTypes.BINARY.value, operator=Operators.DIVISION.value)

        self.ops["y_pred"] = op5
        return op5

    def __create_sgd_ops(self, X_op, y_op, weights_op):
        for i in range(self.iterations):
            # Create ops to get predictions
            y_pred_op = self.__create_predict_ops(X_op=X_op, weights_op=weights_op)

            # Perform y_pred_op - y_op
            sub_op = self.__create_op(name="sub_op", node_type=NodeTypes.MIDDLE.value,
                                      inputs=[y_pred_op.id, y_op.id],
                                      outputs=None, op_type=OpTypes.BINARY.value, operator=Operators.SUBTRACTION.value)
            self.ops["sub_op"] = sub_op

            # Transpose of X_op
            transpose_op = self.__create_op(name="transpose_op", node_type=NodeTypes.MIDDLE.value, inputs=[X_op.id],
                                            outputs=None, op_type=OpTypes.UNARY.value,
                                            operator=Operators.TRANSPOSE.value)
            self.ops["transpose_op"] = transpose_op

            # Matrix multiplication of sub_op and transpose_op
            matmul_op = self.__create_op(name="matmul_op", node_type=NodeTypes.MIDDLE.value,
                                         inputs=[transpose_op.id, sub_op.id], outputs=None,
                                         op_type=OpTypes.BINARY.value, operator=Operators.MATRIX_MULTIPLICATION.value)
            self.ops["matmul_op"] = matmul_op

            # Calculate learning_rate division no_samples
            div_op = self.__create_op(name="div_op", node_type=NodeTypes.MIDDLE.value,
                                      inputs=[self.ops["learning_rate"].id, self.ops['no_samples'].id],
                                      outputs=None, op_type=OpTypes.BINARY.value, operator=Operators.DIVISION.value)

            self.ops["div_op"] = div_op

            # Element-wise-multiplication
            ele_op = self.__create_op(name="ele_op", node_type=NodeTypes.MIDDLE.value,
                                    inputs=[div_op.id, matmul_op.id], outputs=None,
                                    op_type=OpTypes.BINARY.value, operator=Operators.ELEMENT_WISE_MULTIPLICATION.value)
            self.ops["ele_op"] = ele_op

            # Params - params
            updated_weights_op = self.__create_op(name="updated_weights", node_type=NodeTypes.MIDDLE.value,
                                               inputs=[weights_op.id, ele_op.id],
                                               outputs=None, op_type=OpTypes.BINARY.value,
                                               operator=Operators.SUBTRACTION.value)
            self.ops["updated_weights_op"] = updated_weights_op

            # Calculate cost here
            y_pred_op2 = self.__create_predict_ops(X_op=X_op, weights_op=updated_weights_op)
            new_cost_op = self.__create_cost_ops(y_true_op=y_op, y_pred_op=y_pred_op2)

        return new_cost_op, updated_weights_op

    def __create_data_op(self, data_name, data_value, data_type):
        """
        Save data and create an op for it
        """
        data = create_data(self.db, data=data_value, data_type=data_type)
        op = self.__create_op(name=data_name, node_type=NodeTypes.INPUT.value, inputs=None, outputs=[data.id],
                              op_type=OpTypes.UNARY.value, operator=Operators.LINEAR.value,
                              status=OpStatus.COMPUTED.value)
        return op

    def __create_data_ops(self, data_list):
        """
        Create data in database and ops in database for all data
        """
        ops = dict()
        for data_dict in data_list:
            op = self.__create_data_op(data_dict['name'], data_dict['data'], data_dict['type'])
            ops[data_dict['name']] = op

        return ops

    def __create_cost_ops(self, y_true_op, y_pred_op):
        # Cost computation starts here
        # 6. Negation of y
        op6 = self.__create_op(name="op6", node_type=NodeTypes.MIDDLE.value, inputs=[y_true_op.id], outputs=None,
                               op_type=OpTypes.UNARY.value, operator=Operators.NEGATION.value)
        self.ops["op6"] = op6

        # 7. Transpose of op6
        op7 = self.__create_op(name="op7", node_type=NodeTypes.MIDDLE.value, inputs=[op6.id], outputs=None,
                               op_type=OpTypes.UNARY.value, operator=Operators.TRANSPOSE.value)
        self.ops["op7"] = op7

        # 8. Add epsilon to op7
        op8 = self.__create_op(name="op8", node_type=NodeTypes.MIDDLE.value, inputs=[y_pred_op.id, self.ops['epsilon'].id],
                               outputs=None,
                               op_type=OpTypes.BINARY.value, operator=Operators.ADDITION.value)
        self.ops["op8"] = op8

        # 9. Natural Log of op8
        op9 = self.__create_op(name="op9", node_type=NodeTypes.MIDDLE.value, inputs=[op8.id], outputs=None,
                               op_type=OpTypes.UNARY.value, operator=Operators.NATURAL_LOG.value)
        self.ops["op9"] = op9

        # 10. Matrix multiplication of op7 and op9
        op10 = self.__create_op(name="op10", node_type=NodeTypes.MIDDLE.value, inputs=[op7.id, op9.id], outputs=None,
                                op_type=OpTypes.BINARY.value, operator=Operators.MATRIX_MULTIPLICATION.value)
        self.ops["op10"] = op10

        # 11. Subtract y from 1
        op11 = self.__create_op(name="op11", node_type=NodeTypes.MIDDLE.value,
                                inputs=[self.ops['one'].id, y_true_op.id],
                                outputs=None, op_type=OpTypes.BINARY.value, operator=Operators.SUBTRACTION.value)
        self.ops["op11"] = op11

        # 12. Transpose of op11
        op12 = self.__create_op(name="op12", node_type=NodeTypes.MIDDLE.value, inputs=[op11.id], outputs=None,
                                op_type=OpTypes.UNARY.value, operator=Operators.TRANSPOSE.value)
        self.ops["op12"] = op12

        # 13. Subtract y_pred from 1
        op13 = self.__create_op(name="op13", node_type=NodeTypes.MIDDLE.value, inputs=[self.ops['one'].id, y_pred_op.id],
                                outputs=None,
                                op_type=OpTypes.BINARY.value, operator=Operators.SUBTRACTION.value)
        self.ops["op13"] = op13

        # 14. Add epsilon to op13
        op14 = self.__create_op(name="op14", node_type=NodeTypes.MIDDLE.value, inputs=[op13.id, self.ops['epsilon'].id],
                                outputs=None,
                                op_type=OpTypes.BINARY.value, operator=Operators.ADDITION.value)
        self.ops["op14"] = op14

        # 15. Natural Log of op14
        op15 = self.__create_op(name="op15", node_type=NodeTypes.MIDDLE.value, inputs=[op14.id], outputs=None,
                                op_type=OpTypes.UNARY.value, operator=Operators.NATURAL_LOG.value)
        self.ops["op15"] = op15

        # 16. Matrix multiplication of op11 and op15
        op16 = self.__create_op(name="op16", node_type=NodeTypes.MIDDLE.value, inputs=[op12.id, op15.id], outputs=None,
                                op_type=OpTypes.BINARY.value, operator=Operators.MATRIX_MULTIPLICATION.value)
        self.ops["op16"] = op16

        # 17. Subtract op10, 0p16
        op17 = self.__create_op(name="op17", node_type=NodeTypes.MIDDLE.value, inputs=[op10.id, op16.id], outputs=None,
                                op_type=OpTypes.BINARY.value, operator=Operators.SUBTRACTION.value)
        self.ops["op17"] = op17

        # 18. Divide 1 by no.of samples
        op18 = self.__create_op(name="op18", node_type=NodeTypes.MIDDLE.value,
                                inputs=[self.ops['one'].id, self.ops['no_samples'].id], outputs=None,
                                op_type=OpTypes.BINARY.value, operator=Operators.DIVISION.value)
        self.ops["op18"] = op18

        # 19. Multiply op17 and op18
        op19 = self.__create_op(name="cost", node_type=NodeTypes.MIDDLE.value,
                                inputs=[op17.id, op18.id], outputs=None,
                                op_type=OpTypes.BINARY.value, operator=Operators.ELEMENT_WISE_MULTIPLICATION.value)
        self.ops["cost"] = op19

        # Return the cost op
        return op19

    def __create_op(self, name, inputs, outputs, op_type, node_type, operator, status="pending"):
        self.logger.debug("\nCreating an op...")
        self.logger.debug("Values - Name:{}, Inputs:{}, Outputs:{}, Status:{}, Node Type:{}, Op Type:{}, "
                          "Operator:{}".format(name, inputs, outputs, status,
                                               node_type, op_type, operator))
        # List of op ids
        inputs = json.dumps(inputs)

        # List of data ids
        outputs = json.dumps(outputs)

        # Create an op and save it
        op = Op()
        op.name = name
        op.graph_id = self.graph.id
        op.node_type = node_type
        op.inputs = inputs
        op.outputs = outputs
        op.op_type = op_type
        op.operator = operator
        op.status = status

        op = self.db.add(op)

        self.logger.debug("Op created:{}".format(op.id))
        # Return database op
        return op

# def sigmoid(op_id, graph_id, db):
#     # Numpy - 1 / (1 + np.exp(-x))
#
#     # Operations
#     # 1. Negate x
#     op1_id = create_op(graph_id=graph_id, db=db, node_type=NodeTypes.MIDDLE.value, inputs=[op_id], outputs=None,
#                        op_type=OpTypes.UNARY.value, operator=Operators.NEGATION.value)
#
#     # 2. Exponential of the output of step 1
#     op2_id = create_op(graph_id=graph_id, db=db, node_type=NodeTypes.MIDDLE.value, inputs=[op1_id], outputs=None,
#                        op_type=OpTypes.UNARY.value, operator=Operators.EXPONENTIAL.value)
#
#     # Create data and op for 1
#     d_id = create_data(db, data=1)
#     op3_id = create_op(graph_id=graph_id, db=db, node_type=NodeTypes.MIDDLE.value, inputs=None, outputs=[d_id],
#                        op_type=OpTypes.UNARY.value, operator=Operators.LINEAR.value)
#
#     # 3. Add numeric 1 to the output of step 2
#     op4_id = create_op(graph_id=graph_id, db=db, node_type=NodeTypes.MIDDLE.value, inputs=[op2_id, op3_id],
#                        outputs=None,
#                        op_type=OpTypes.BINARY.value, operator=Operators.ADDITION.value)
#
#     # 4. Divide numeric 1 by the output of step 3
#     op5_id = create_op(graph_id=graph_id, db=db, node_type=NodeTypes.MIDDLE.value, inputs=[op3_id, op4_id],
#                        outputs=None,
#                        op_type=OpTypes.BINARY.value, operator=Operators.DIVISION.value)
#
#     return op5_id


# # Softmax activation function
# def softmax(x, raven=False):
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum(axis=0)
#

# Calculate cost value(loss value)
# def calculate_cost(y, y_pred, no_samples, epsilon, raven=False):
#     """
#     Numpy formula (1/no_samples)*(((-y).T @ np.log(y_pred))-((1-y).T @ np.log(1-y_pred)))
#     """
#
#     # Operations
#     # 1. Negate y
#     if raven:
#         # Raven
#         # Coming soon
#         y1 = None
#     else:
#         # Numpy
#         y1 = -y  # -1 * y - Alternate method
#
#     # 2. Transpose y
#     if raven:
#         # Raven
#         # Coming soon
#         y2 = None
#     else:
#         # Numpy
#         y2 = np.transpose(y1)  # y1.T - Alternate method
#
#     # 3. Natural log of y_pred
#     if raven:
#         # Raven
#         # Coming soon
#         y_pred_log = None
#     else:
#         # Numpy
#         y_pred_log = np.log(y_pred + epsilon)
#
#     # 4. Calculate the dot product of the output of step 2 and step 3
#     if raven:
#         # Raven
#         # Coming soon
#         o1 = None
#     else:
#         # Numpy
#         o1 = np.dot(y2, y_pred_log)
#
#     # 5. Subtract y from 1
#     if raven:
#         # Raven
#         # Coming soon
#         y3 = None
#     else:
#         # Numpy
#         y3 = 1 - y
#
#     # 6. Transpose the output of step 5
#     if raven:
#         # Raven
#         # Coming soon
#         y4 = None
#     else:
#         # Numpy
#         y4 = np.transpose(y3)  # y3.T - Alternate method
#
#     # 7. Subtract y_pred from 1
#     if raven:
#         # Raven
#         # Coming soon
#         y_pred1 = None
#     else:
#         # Numpy
#         y_pred1 = 1 - y_pred
#
#     # 8. Natural log of the output of step 7
#     if raven:
#         # Raven
#         # Coming soon
#         y_pred1_log = None
#     else:
#         # Numpy
#         y_pred1_log = np.log(y_pred1 + epsilon)
#
#     # 9. Dot product of the output of step 6 and step 8
#     if raven:
#         # Raven
#         # Coming soon
#         o2 = None
#     else:
#         # Numpy
#         o2 = np.dot(y4, y_pred1_log)
#
#     # 10. Subtract 5 and 9
#     if raven:
#         # Raven
#         # Coming soon
#         o3 = None
#     else:
#         # Numpy
#         o3 = o1 - o2
#
#     # 11. Divide 1 and number of samples
#     if raven:
#         # Raven
#         # Coming soon
#         n_d = None
#     else:
#         # Numpy
#         n_d = 1 / no_samples
#
#     # 12. Multiply 8 and 9
#     if raven:
#         # Raven
#         # Coming soon
#         final_output = None
#     else:
#         # Numpy
#         final_output = n_d * o3
#
#     return final_output


# def predict(op1_id, op2_id, graph_id, db):
#     # 1. Create an operation to multiply X and params
#     op3_id = create_op(graph_id=graph_id, node_type=NodeTypes.MIDDLE.value, inputs=[op1_id, op2_id], outputs=None,
#                        op_type=OpTypes.BINARY.value, operator=Operators.MATRIX_MULTIPLICATION.value, db=db)
#
#     print("Op3 id", op3_id)
#
#     op4_id = sigmoid(op3_id, graph_id, db)

# def create_op(graph_id, db, inputs, outputs, op_type, node_type, operator, status="pending"):
#     print("Op:", graph_id, node_type, inputs, outputs, op_type, operator)
#
#     # List of op ids
#     inputs = json.dumps(inputs)
#
#     # List of data ids
#     outputs = json.dumps(outputs)
#
#     # Create an op and save it
#     op = Op()
#     op.graph_id = graph_id
#     op.node_type = node_type
#     op.inputs = inputs
#     op.outputs = outputs
#     op.op_type = op_type
#     op.operator = operator
#     op.status = status
#
#     op = db.add(op)
#
#     print("Op created:", op.id)
#     return op.id
