import json
import logging
import logging.handlers
import pickle

from common.db_manager import DBManager, Op, NodeTypes, Operators, OpTypes, OpStatus, Data
from .constants import RAVOP_LOG_FILE


class GraphManager(object):
    def __init__(self, graph_id=None):
        self.graph_id = graph_id

        # Create common client
        self.db = DBManager()

        self.logger = None
        self.__setup_logger()

        self.graph = None
        self.__create_or_get_graph()

    def __create_or_get_graph(self):
        # Create or get a graph in database
        if self.graph_id is None:
            self.graph = self.db.create_graph()
        else:
            self.graph = self.db.get_graph(self.graph_id)

            if self.graph is None:
                self.graph = self.db.create_graph()

    def __setup_logger(self):
        # Set up a specific logger with our desired output level
        self.logger = logging.getLogger(GraphManager.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

        # Add the log message handler to the logger
        handler = logging.handlers.RotatingFileHandler(RAVOP_LOG_FILE)

        self.logger.addHandler(handler)

    def clear_graph(self):
        self.db.delete_graph_ops(self.graph_id)

    def print_ops(self):
        self.logger.debug("\nOps")
        ops = self.db.get_graph_ops(graph_id=self.graph_id)
        for op in ops:
            self.logger.debug("Values - Name:{}, Op Id:{}, Inputs:{}, Outputs:{}, Status:{}, Node Type:{}, Op Type:{}, "
                              "Operator:{}".format(op.name, op.id, op.inputs, op.outputs, op.status,
                                                   op.node_type, op.op_type, op.operator))

    def create_data_op(self, graph_id, data_name, data_value, data_type):
        """
        Save data and create an op for it
        """
        data = self.db.create_data_complete(data=data_value, data_type=data_type)
        op = self.create_op(graph_id=graph_id, name=data_name, node_type=NodeTypes.INPUT.value,
                            inputs=None, outputs=[data.id],
                            op_type=OpTypes.UNARY.value, operator=Operators.LINEAR.value,
                            status=OpStatus.COMPUTED.value)
        return op

    def create_data_ops(self, graph_id, data_list):
        """
        Create data in common and ops in common for all data
        """
        ops = dict()
        for data_dict in data_list:
            op = self.create_data_op(graph_id, data_dict['name'], data_dict['data'], data_dict['type'])
            ops[data_dict['name']] = op

        return ops

    def create_op(self, graph_id, name, inputs, outputs, op_type, node_type, operator, status="pending"):
        self.logger.debug("\nCreating an op...")
        self.logger.debug("Values - Name:{}, Inputs:{}, Outputs:{}, Status:{}, Node Type:{}, Op Type:{}, "
                          "Operator:{}".format(name, inputs, outputs, status,
                                               node_type, op_type, operator))
        # List of op ids
        inputs = json.dumps(inputs)

        # List of data ids
        outputs = json.dumps(outputs)

        op = self.db.create_op(name=name,
                               graph_id=graph_id, node_type=node_type, inputs=inputs,
                               outputs=outputs, op_type=op_type, operator=operator, status=status
                               )

        self.logger.debug("Op created:{}".format(op.id))
        # Return common op
        return op

    def get_op_outputs(self, op_id):
        self.db = DBManager()
        print(op_id)
        print(self.db.get_op(op_id).outputs, self.db.get_op(op_id).status)
        data_id = json.loads(self.db.get_op(op_id).outputs)[0]
        data = self.db.get_data(data_id=data_id)
        file_path = data.file_path

        # print("File path:", file_path)

        with open(file_path, "rb") as f:
            a = json.load(f)
            # print("Data:", a, type(a), data.type)

            if data.type == "integer":
                return a
            elif data.type == "ndarray":
                return a.tolist()
            else:
                return a

    def get_op_status(self, op_id):
        return self.db.get_op_status(op_id)
