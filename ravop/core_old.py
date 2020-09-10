import json
import os

import numpy as np

from common.constants import DATA_FILES_PATH
from common.db_manager import NodeTypes, OpTypes, Operators, DBManager, OpStatus

"""
Op Class to create and get op
"""


class Op(object):
    def __init__(self, id=None,
                 operator=None, inputs=None, outputs=None, **kwargs):
        self._id = id
        self.db = DBManager.Instance()

        if self._id is not None:
            db_op = self.db.get_op(op_id=self._id)
            if db_op is None:
                raise Exception("Invalid op id")
            else:
                self._id = db_op.id
        else:
            if (inputs is not None or outputs is not None) and operator is not None:
                db_op = self.create(operator=operator, inputs=inputs, outputs=outputs, **kwargs)
                self._id = db_op.id
            else:
                raise Exception("Invalid parameters")

    def create(self, operator, inputs=None, outputs=None, **kwargs):
        if (inputs is not None or outputs is not None) and operator is not None:
            # Figure out node type
            if inputs is None and outputs is not None:
                node_type = NodeTypes.INPUT.value
            elif inputs is not None and outputs is None:
                node_type = NodeTypes.MIDDLE.value
            else:
                raise Exception("Invalid node type")

            if inputs is not None:
                if len(inputs) == 1:
                    op_type = OpTypes.UNARY.value
                elif len(inputs) == 2:
                    op_type = OpTypes.BINARY.value
                else:
                    raise Exception("Invalid number of inputs")
            else:
                op_type = OpTypes.OTHER.value

            if outputs is None:
                status = OpStatus.PENDING.value
            else:
                status = OpStatus.COMPUTED.value

            inputs = json.dumps(inputs)
            outputs = json.dumps(outputs)

            op = self.db.create_op(name=kwargs.get("name", None),
                                   graph_id=kwargs.get("graph_id", None),
                                   node_type=node_type,
                                   inputs=inputs,
                                   outputs=outputs,
                                   op_type=op_type,
                                   operator=operator,
                                   status=status)
            return op
        else:
            raise Exception("Invalid parameters")

    def add(self, op, **kwargs):
        return self.__create_math_op(self, op, Operators.ADDITION.value, **kwargs)

    def sub(self, op, **kwargs):
        return self.__create_math_op(self, op, Operators.SUBTRACTION.value, **kwargs)

    def matmul(self, op, **kwargs):
        return self.__create_math_op(self, op, Operators.MATRIX_MULTIPLICATION.value, **kwargs)

    def div(self, op, **kwargs):
        return self.__create_math_op(self, op, Operators.DIVISION.value, **kwargs)

    def elemul(self, op, **kwargs):
        return self.__create_math_op(self, op, Operators.ELEMENT_WISE_MULTIPLICATION.value, **kwargs)

    def neg(self, **kwargs):
        return self.__create_math_op2(self, Operators.NEGATION.value, **kwargs)

    def exp(self, **kwargs):
        return self.__create_math_op2(self, Operators.EXPONENTIAL.value, **kwargs)

    def trans(self, **kwargs):
        return self.__create_math_op2(self, Operators.TRANSPOSE.value, **kwargs)

    def natlog(self, **kwargs):
        return self.__create_math_op2(self, Operators.NATURAL_LOG.value, **kwargs)

    def matsum(self, **kwargs):
        return self.__create_math_op2(self, Operators.MATRIX_SUM.value, **kwargs)

    def linear(self, **kwargs):
        return self.__create_math_op2(self, Operators.LINEAR.value, **kwargs)

    def __create_math_op(self, op1, op2, operator, **kwargs):
        if op1 is None or op2 is None:
            raise Exception("Null Op")

        op = self.db.create_op(name=kwargs.get('name', None),
                               graph_id=kwargs.get('graph_id', None),
                               node_type=NodeTypes.MIDDLE.value,
                               inputs=json.dumps([op1.id, op2.id]),
                               outputs=json.dumps(None),
                               op_type=OpTypes.BINARY.value,
                               operator=operator,
                               status=OpStatus.PENDING.value)
        return Op(id=op.id)

    def __create_math_op2(self, op1, operator, **kwargs):
        if op1 is None:
            raise Exception("Null Op")

        op = self.db.create_op(name=kwargs.get('name', None),
                               graph_id=kwargs.get('graph_id', None),
                               node_type=NodeTypes.MIDDLE.value,
                               inputs=json.dumps([op1.id]),
                               outputs=json.dumps(None),
                               op_type=OpTypes.UNARY.value,
                               operator=operator,
                               status=OpStatus.PENDING.value)
        return Op(id=op.id)

    def __build_op(self):
        return self.db.get_op(op_id=self.id)

    @property
    def output(self):
        op = self.__build_op()
        if op.outputs is None or op.outputs == "null":
            return None

        data_id = json.loads(op.outputs)[0]
        data = Data(id=data_id)
        return data.value

    @property
    def output_dtype(self):
        op = self.__build_op()
        if op.outputs is None or op.outputs == "null":
            return None

        data_id = json.loads(op.outputs)[0]
        data = Data(id=data_id)
        return data.dtype

    @property
    def id(self):
        return self._id

    @property
    def status(self):
        if self._id is None:
            raise Exception("Invalid op")

        return self.db.get_op_status(self.id)

    def add_to_graph(self, graph):
        op = self.__build_op()
        self.db.update_op(op.id, graph_id=graph.id)

    def __str__(self):
        op = self.__build_op()
        return "Op:\nId:{}\nType:{}\nOperator:{}\nOutput:{}\nStatus:{}\n".format(self.id, op.op_type,
                                                                                 op.operator, self.output,
                                                                                 self.status)


class Scalar(Op):
    def __init__(self, value, **kwargs):
        # 1. Store it in a file
        # 2. Create data object
        # 3. Create data op

        self.db = DBManager.Instance()

        # Find dtype
        self._dtype = None
        self.__find_dtype(value)

        data = Data(value=value, dtype=self.dtype)
        super().__init__(operator=Operators.LINEAR.value, inputs=None, outputs=[data.id], **kwargs)

    def __find_dtype(self, x):
        if type(x).__name__ == "str":
            x = json.loads(x)
            if type(x).__name__ == "int":
                self._dtype = "int"
            elif type(x).__name__ == "float":
                self._dtype = "float"
            elif type(x).__name__ == "list":
                raise Exception("Invalid data")
            else:
                raise Exception("Invalid value")

        elif type(x).__name__ == "int":
            self._dtype = "int"
        elif type(x).__name__ == "float":
            self._dtype = "float"
        elif type(x).__name__ == "list":
            raise Exception("Invalid data")

    @property
    def dtype(self):
        return self._dtype

    def __str__(self):
        return "Scalar Op:\nId:{}\nOutput:{}\nStatus:{}\nDtype:{}\n".format(self.id, self.output,
                                                                            self.status, self.dtype)


class Tensor(Op):
    """
    It supports:
    1. list
    2. ndarray
    3. string(list)
    """

    def __init__(self, value, **kwargs):
        # 1. Store it in a file
        # 2. Create data object
        # 3. Create data op
        """
        Kwargs
        1.db
        2.name
        3.graph
        """
        self.db = DBManager.Instance()

        if type(value).__name__ == "list":
            value = np.array(value)
        elif type(value).__name__ == "str":
            x = json.loads(value)
            if type(x).__name__ == "list":
                value = np.array(value)

        self._shape = value.shape

        data = Data(value=value, dtype="ndarray")
        super().__init__(operator=Operators.LINEAR.value, inputs=None, outputs=[data.id], **kwargs)

    @property
    def dtype(self):
        return "ndarray"

    @property
    def shape(self):
        return self._shape

    def __str__(self):
        return "Tensor Op:\nId:{}\nOutput:{}\nStatus:{}\nDtype:{}\n".format(self.id, self.output,
                                                                            self.status, self.dtype)


class Data(object):
    def __init__(self, id=None, value=None, dtype=None, **kwargs):
        self._id = id
        self.db = DBManager.Instance()

        if self._id is not None:
            data_db = self.db.get_data(data_id=self._id)
            if data_db is None:
                raise Exception("Invalid data id")

        elif value is not None and dtype is not None:

            if type(value).__name__ == "list":
                value = np.array(value)
            elif type(value).__name__ == "str":
                x = json.loads(value)
                if type(x).__name__ == "list":
                    value = np.array(value)

            data_db = self.__create(value=value, dtype=dtype)
            if data_db is None:
                raise Exception("Unable to create data")
            else:
                self._id = data_db.id

    def __create(self, value, dtype):
        data = self.db.create_data(type=dtype)

        if dtype == "ndarray":
            file_path = os.path.join(DATA_FILES_PATH, "data_{}.pkl".format(data.id))
            if os.path.exists(file_path):
                os.remove(file_path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            value.dump(file_path)

            # Update file path
            self.db.update_data(data.id, file_path=file_path)
        elif dtype in ["int", "float"]:
            self.db.update_data(data.id, value=value)

        return data

    def __build_data(self):
        return self.db.get_data(data_id=self._id)

    @property
    def value(self):
        data = self.__build_data()

        if self.dtype == "ndarray":
            file_path = data.file_path

            value = np.load(file_path, allow_pickle=True)
            return value
        elif self.dtype in ["int", "float"]:
            if self.dtype == "int":
                return int(data.value)
            elif self.dtype == "float":
                return float(data.value)

    @property
    def id(self):
        return self._id

    @property
    def dtype(self):
        data = self.__build_data()
        return data.type

    def __str__(self):
        return "Data:\nId:{}\nDtype:{}\n".format(self.id, self.dtype)


class Graph(object):
    def __init__(self, id=None, **kwargs):
        self._id = id
        self.db = DBManager.Instance()

        if self._id is None:
            # Create a new graph
            graph_db = self.db.create_graph()
            self._id = graph_db.id
        else:
            graph_db = self.db.get_graph(graph_id=self._id)

        if graph_db is None:
            raise Exception("Invalid graph id")
        else:
            self._id = graph_db.id

        self.ops = list()

    def __build_graph(self):
        return self.db.get_graph(graph_id=self._id)

    def add(self, op):
        self.ops.append(op)
        graph = self.__build_graph()
        op.add_to_graph(graph)

    @property
    def id(self):
        return self._id

    @property
    def status(self):
        graph = self.__build_graph()
        return graph.status

    def clean(self):
        graph = self.__build_graph()
        self.db.delete_graph_ops(graph.id)

    def __str__(self):
        return "Graph:\nId:{}\nStatus:{}\n".format(self.id, self.status)
