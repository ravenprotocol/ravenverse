import json
import logging
import logging.handlers

from database.db_manager import DBManager, Graph, Op, NodeTypes, Operators, OpTypes
from database.db_utils import create_graph
from op_manager import OpManager
from sockets.socket_client import SocketClient
from settings import LOG_FILE
import numpy as np


class OpBase(object):
    def __init__(self, graph_id=None, socket_client=None):
        # Create database client
        self.db = DBManager()

        self.db.create_tables()

        self.op_manager = OpManager()

        self.socket_client = socket_client

        # Create a graph in database
        if graph_id is None:
            self.graph = create_graph(self.db)
        else:
            self.graph = self.db.session.query(Graph).get(graph_id)

            if self.graph is None:
                self.graph = create_graph(self.db)

        # Set up a specific logger with our desired output level
        self.logger = logging.getLogger(OpBase.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

        # Add the log message handler to the logger
        handler = logging.handlers.RotatingFileHandler(
            LOG_FILE)

        self.logger.addHandler(handler)

        self.output_op_id = None
        self.op_status = "not started"

    def compute(self, data_list, operator, op_type):
        print("Computing...")

        # create ops for data
        data_ops = self.op_manager.create_data_ops(graph_id=self.graph.id, data_list=data_list)

        inputs = []
        for key, data_op in data_ops.items():
            inputs.append(data_op.id)

        output_op = self.op_manager.create_op(graph_id=self.graph.id, name="ass", inputs=inputs, outputs=[],
                                                   op_type=op_type,
                                                   node_type=NodeTypes.MIDDLE.value, operator=operator)

        self.socket_client.emit("update_server", data=None, namespace="/ravop")

        self.output_op_id = output_op.id

        self.op_status = "computing"

        while self.op_status != "computed":
            self.op_status = self.db.get_op_status(self.output_op_id)

    def get_result(self):
        if self.db.get_op_status(self.output_op_id) == "computing":
            print("Computing...")
            return None
        else:
            status = self.db.get_op_status(self.output_op_id)
            result = self.op_manager.get_op_outputs(self.output_op_id)
            print("Result:", result)
            return result


def get_operator_name(operator_number):
    operators = [e.value for e in Operators]
    return operators[operator_number - 1]


if __name__ == '__main__':
    socket_client = SocketClient().connect()

    i = 0
    while True:
        op_base = OpBase(graph_id=i, socket_client=socket_client)
        i += 1

        # Create a graph here
        print("\n======================")
        print("Select Operator:")
        print("1. Matrix Multiplication")
        print("2. Addition")
        print("3. Subtraction")
        print("4. Element-wise-multiplication")
        print("5. Division")
        print("6. Linear")
        print("7. Negation")
        print("8. Exponential")
        print("9. Transpose")
        print("10. Natural Log")

        print("\n")

        operator_number = input("Enter operator number here: ")

        while operator_number == "":
            operator_number = input("Enter operator number here: ")

        operator_number = int(operator_number)

        if operator_number > 10 or operator_number < 1:
            print("Enter a valid operator number")
            continue

        if operator_number <= 5:
            operand1 = input("\nEnter the first value(Scalar, Array or Matrix): ")

            try:
                operand1 = json.loads(operand1)
                operand1 = np.array(operand1)
                operand_type = "ndarray"
            except Exception as e:
                print("Error:", e)
                try:
                    operand1 = int(operand1)
                    operand_type = "integer"
                except Exception as e:
                    print("Error:", e)
                    print("Unsupported operand type")
                    continue

            operand2 = input("\nEnter the second value(Scalar, Array or Matrix): ")

            try:
                operand2 = json.loads(operand2)
                operand2 = np.array(operand2)
                operand_type = "ndarray"
            except Exception as e:
                print("Error:", e)
                try:
                    operand2 = int(operand2)
                    operand_type = "integer"
                except Exception as e:
                    print("Error:", e)
                    print("Unsupported operand type")
                    continue

            # Create a operation
            data_list = [{"name": "x", "data": operand1, "type": operand_type},
                         {"name": "y", "data": operand2, "type": operand_type}]

            op_base.compute(data_list=data_list, operator=get_operator_name(operator_number),
                            op_type=OpTypes.BINARY.value)

            op_base.get_result()

        else:
            operand1 = input("\nEnter the value(Scalar, Array or Matrix): ")

            try:
                operand1 = json.loads(operand1)
                operand1 = np.array(operand1)
                operand_type = "ndarray"
            except Exception as e:
                print("Error:", e)
                try:
                    operand1 = int(operand1)
                    operand_type = "integer"
                except Exception as e:
                    print("Error:", e)
                    print("Unsupported operand type")
                    continue

            # Create a operation
            data_list = [{"name": "x", "data": operand1, "type": operand_type}]

            op_base.compute(data_list=data_list, operator=get_operator_name(operator_number),
                            op_type=OpTypes.UNARY.value)

            op_base.get_result()
