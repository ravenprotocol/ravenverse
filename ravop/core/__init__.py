import json
import os

import numpy as np

from common import db
from common.constants import DATA_FILES_PATH
from common.db_manager import NodeTypes, OpTypes, Operators, OpStatus
from ravop import globals as g


class Op(object):
    def __init__(self, id=None,
                 operator=None, inputs=None, outputs=None, **kwargs):
        self._op_db = None

        if id is not None:
            self._op_db = db.get_op(op_id=id)
            if self._op_db is None:
                raise Exception("Invalid op id")
        else:
            if (inputs is not None or outputs is not None) and operator is not None:
                self._op_db = self.create(operator=operator, inputs=inputs, outputs=outputs, **kwargs)
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

            op = db.create_op(name=kwargs.get("name", None),
                              graph_id=g.graph_id,
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

        op = db.create_op(name=kwargs.get('name', None),
                          graph_id=g.graph_id,
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

        op = db.create_op(name=kwargs.get('name', None),
                          graph_id=g.graph_id,
                          node_type=NodeTypes.MIDDLE.value,
                          inputs=json.dumps([op1.id]),
                          outputs=json.dumps(None),
                          op_type=OpTypes.UNARY.value,
                          operator=operator,
                          status=OpStatus.PENDING.value)
        return Op(id=op.id)

    @property
    def output(self):
        self._op_db = db.refresh(self._op_db)
        if self._op_db.outputs is None or self._op_db.outputs == "null":
            return None

        data_id = json.loads(self._op_db.outputs)[0]
        data = Data(id=data_id)
        return data.value

    @property
    def output_dtype(self):
        self._op_db = db.refresh(self._op_db)
        if self._op_db.outputs is None or self._op_db.outputs == "null":
            return None

        data_id = json.loads(self._op_db.outputs)[0]
        data = Data(id=data_id)
        return data.dtype

    @property
    def id(self):
        return self._op_db.id

    @property
    def status(self):
        self._op_db = db.refresh(self._op_db)
        return self._op_db.status

    def __str__(self):
        return "Op:\nId:{}\nName:{}\nType:{}\nOperator:{}\nOutput:{}\nStatus:{}\n".format(self.id, self._op_db.name,
                                                                                          self._op_db.op_type,
                                                                                          self._op_db.operator,
                                                                                          self.output,
                                                                                          self.status)


class Scalar(Op):
    def __init__(self, value, **kwargs):
        # 1. Store it in a file
        # 2. Create data object
        # 3. Create data op

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
        self._data_db = None

        if id is not None:
            self._data_db = db.get_data(data_id=id)
            if self._data_db is None:
                raise Exception("Invalid data id")

        elif value is not None and dtype is not None:

            if type(value).__name__ == "list":
                value = np.array(value)
            elif type(value).__name__ == "str":
                x = json.loads(value)
                if type(x).__name__ == "list":
                    value = np.array(value)

            self._data_db = self.__create(value=value, dtype=dtype)
            if self._data_db is None:
                raise Exception("Unable to create data")

    def __create(self, value, dtype):
        data = db.create_data(type=dtype)

        if dtype == "ndarray":
            file_path = os.path.join(DATA_FILES_PATH, "data_{}.pkl".format(data.id))
            if os.path.exists(file_path):
                os.remove(file_path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            value.dump(file_path)

            # Update file path
            db.update_data(data, file_path=file_path)
        elif dtype in ["int", "float"]:
            db.update_data(data, value=value)

        return data

    @property
    def value(self):
        self._data_db = db.refresh(self._data_db)
        if self.dtype == "ndarray":
            file_path = self._data_db.file_path
            value = np.load(file_path, allow_pickle=True)
            return value
        elif self.dtype in ["int", "float"]:
            if self.dtype == "int":
                return int(self._data_db.value)
            elif self.dtype == "float":
                return float(self._data_db.value)

    @property
    def id(self):
        return self._data_db.id

    @property
    def dtype(self):
        self._data_db = db.refresh(self._data_db)
        return self._data_db.type

    def __str__(self):
        return "Data:\nId:{}\nDtype:{}\n".format(self.id, self.dtype)


class Graph(object):
    """A class to represent a graph object"""

    def __init__(self, id=None, **kwargs):
        if id is None and g.graph_id is None:
            # Create a new graph
            self._graph_db = db.create_graph()
            g.graph_id = self._graph_db.id
        elif id is not None:
            # Get an existing graph
            self._graph_db = db.get_graph(graph_id=id)
        elif g.graph_id is not None:
            # Get an existing graph
            self._graph_db = db.get_graph(graph_id=g.graph_id)

        # Raise an exception if there is no graph created
        if self._graph_db is None:
            raise Exception("Invalid graph id")

    def add(self, op):
        """Add an op to the graph"""
        op.add_to_graph(self._graph_db)

    @property
    def id(self):
        return self._graph_db.id

    @property
    def status(self):
        self._graph_db = db.refresh(self._graph_db)
        return self._graph_db.status

    @property
    def progress(self):
        """Get the progress"""
        stats = self.get_op_stats()
        progress = ((stats["computed_ops"]+stats["computing_ops"]+stats["failed_ops"])/stats["total_ops"])*100
        return progress

    def get_op_stats(self):
        """Get stats of all ops"""
        ops = db.get_graph_ops(graph_id=self.id)

        pending_ops = 0
        computed_ops = 0
        computing_ops = 0
        failed_ops = 0

        for op in ops:
            if op.status == "pending":
                pending_ops += 1
            elif op.status == "computed":
                computed_ops += 1
            elif op.status == "computing":
                computing_ops += 1
            elif op.status == "failed":
                failed_ops += 1

        total_ops = len(ops)
        return {"total_ops": total_ops, "pending_ops": pending_ops,
                "computing_ops": computing_ops, "computed_ops": computed_ops,
                "failed_ops": failed_ops}

    def clean(self):
        db.delete_graph_ops(self._graph_db.id)

    @property
    def ops(self):
        ops = db.get_graph_ops(self.id)
        return [Op(id=op.id) for op in ops]

    def print_ops(self):
        """Print ops"""
        for op in self.ops:
            print(op)

    def get_ops_by_name(self, op_name, graph_id=None):
        ops = db.get_ops_by_name(op_name=op_name, graph_id=graph_id)
        return [Op(id=op.id) for op in ops]

    def __str__(self):
        return "Graph:\nId:{}\nStatus:{}\n".format(self.id, self.status)


"""
Functional Interface
"""


def add(op1, op2, **kwargs):
    return __create_math_op(op1, op2, Operators.ADDITION.value, **kwargs)


def sub(op1, op2, **kwargs):
    return __create_math_op(op1, op2, Operators.SUBTRACTION.value, **kwargs)


def matmul(op1, op2, **kwargs):
    return __create_math_op(op1, op2, Operators.MATRIX_MULTIPLICATION.value, **kwargs)


def div(op1, op2, **kwargs):
    return __create_math_op(op1, op2, Operators.DIVISION.value, **kwargs)


def elemul(op1, op2, **kwargs):
    return __create_math_op(op1, op2, Operators.ELEMENT_WISE_MULTIPLICATION.value, **kwargs)


def neg(op, **kwargs):
    return __create_math_op2(op, Operators.NEGATION.value, **kwargs)


def exp(op, **kwargs):
    return __create_math_op2(op, Operators.EXPONENTIAL.value, **kwargs)


def trans(op, **kwargs):
    return __create_math_op2(op, Operators.TRANSPOSE.value, **kwargs)


def natlog(op, **kwargs):
    return __create_math_op2(op, Operators.NATURAL_LOG.value, **kwargs)


def matsum(op, **kwargs):
    return __create_math_op2(op, Operators.MATRIX_SUM.value, **kwargs)


def linear(op, **kwargs):
    return __create_math_op2(op, Operators.LINEAR.value, **kwargs)


def __create_math_op(op1, op2, operator, **kwargs):
    if op1 is None or op2 is None:
        raise Exception("Null Op")

    op = db.create_op(name=kwargs.get('name', None),
                      graph_id=g.graph_id,
                      node_type=NodeTypes.MIDDLE.value,
                      inputs=json.dumps([op1.id, op2.id]),
                      outputs=json.dumps(None),
                      op_type=OpTypes.BINARY.value,
                      operator=operator,
                      status=OpStatus.PENDING.value)
    return Op(id=op.id)


def __create_math_op2(op1, operator, **kwargs):
    if op1 is None:
        raise Exception("Null Op")

    op = db.create_op(name=kwargs.get('name', None),
                      graph_id=g.graph_id,
                      node_type=NodeTypes.MIDDLE.value,
                      inputs=json.dumps([op1.id]),
                      outputs=json.dumps(None),
                      op_type=OpTypes.UNARY.value,
                      operator=operator,
                      status=OpStatus.PENDING.value)
    return Op(id=op.id)
