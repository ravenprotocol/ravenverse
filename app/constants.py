from enum import Enum


# Validation enums
class DataTypes(Enum):
    scalar = "scalar"
    ndarray = "ndarray"


class ClientTypes(Enum):
    RAVOP = "ravop"
    RAVJS = "ravjs"


class OpTypes(Enum):
    UNARY = "unary"
    BINARY = "binary"
    OTHER = "other"


class NodeTypes(Enum):
    INPUT = "input"
    MIDDLE = "middle"
    OUTPUT = "output"


class Operators(Enum):
    MATRIX_MULTIPLICATION = "matrix_multiplication"
    ADDITION = "addition"
    SUBTRACTION = "subtraction"
    ELEMENT_WISE_MULTIPLICATION = "element_wise_multiplication"
    DIVISION = "division"
    LINEAR = "linear"
    NEGATION = "negation"
    EXPONENTIAL = "exponential"
    TRANSPOSE = "transpose"
    NATURAL_LOG = "natural_log"


class OpStatus(Enum):
    PENDING = "pending"
    COMPUTING = "computing"
    COMPUTED = "computed"
    FAILED = "failed"


class ClientStatus(Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
