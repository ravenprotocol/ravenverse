import logging
import logging.handlers

from common.db_manager import DBManager, NodeTypes
from ravop.graph_manager import GraphManager
from ravop.constants import RAVOP_LOG_FILE
from ravop.socket_client import SocketClient


class Core(object):
    def __init__(self, graph_id=None):
        self.LOG = Core.__class__.__name__
        self.graph_id = graph_id

        # Create common client
        self.db = DBManager()
        self.graph_manager = GraphManager(graph_id=graph_id)
        self.socket_client = SocketClient().connect()

        self.logger = None
        self.__setup_logger()

        self.output_op_id = None
        self.op_status = "not started"

    def __setup_logger(self):
        # Set up a specific logger with our desired output level
        self.logger = logging.getLogger(self.LOG)
        self.logger.setLevel(logging.DEBUG)

        # Add the log message handler to the logger
        handler = logging.handlers.RotatingFileHandler(RAVOP_LOG_FILE)

        self.logger.addHandler(handler)

    def compute(self, data_list, operator, op_type):
        self.logger.debug("{}:Computing...".format(self.LOG))

        # create ops for data
        data_ops = self.graph_manager.create_data_ops(graph_id=self.graph_id, data_list=data_list)

        inputs = []
        for key, data_op in data_ops.items():
            inputs.append(data_op.id)

        output_op = self.graph_manager.create_op(graph_id=self.graph_id, name="ass", inputs=inputs, outputs=[],
                                                 op_type=op_type,
                                                 node_type=NodeTypes.MIDDLE.value, operator=operator)

        self.socket_client.emit("update_server", data=None, namespace="/ravop")

        self.output_op_id = output_op.id

        self.op_status = "computing"

        while self.op_status != "computed":
            self.db = DBManager()
            self.op_status = self.db.get_op_status(self.output_op_id)

    def get_result(self):
        if self.db.get_op_status(self.output_op_id) == "computing":
            print("Computing...")
            return None
        else:
            status = self.db.get_op_status(self.output_op_id)
            result = self.graph_manager.get_op_outputs(self.output_op_id)
            print("Result:{}".format(result))
            return result
