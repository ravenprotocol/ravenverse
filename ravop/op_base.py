import logging

from common.db_manager import Graph, DBManager, NodeTypes, create_graph
from .op_manager import OpManager
from .constants import RAVOP_LOG_FILE


class OpBase(object):
    def __init__(self, graph_id=None, socket_client=None):
        # Create common client
        self.db = DBManager()

        # self.db.create_tables()

        self.op_manager = OpManager()

        self.socket_client = socket_client

        # Create a graph in common
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
        handler = logging.handlers.RotatingFileHandler(RAVOP_LOG_FILE)

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
