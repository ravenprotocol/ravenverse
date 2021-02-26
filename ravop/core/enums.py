from enum import Enum


class OpTypes(Enum):
    UNARY = "unary"
    BINARY = "binary"
    OTHER = "other"


class NodeTypes(Enum):
    INPUT = "input"
    MIDDLE = "middle"
    OUTPUT = "output"


class Operators(Enum):
    # Arithmetic
    LINEAR = "linear"
    ADDITION = "addition"
    SUBTRACTION = "subtraction"
    MULTIPLICATION = "multiplication"
    DIVISION = "division"
    NEGATION = "negation"
    EXPONENTIAL = "exponential"
    NATURAL_LOG = "natural_log"
    POWER = "power"
    SQUARE = "square"
    CUBE = "cube"
    SQUARE_ROOT = "square_root"
    CUBE_ROOT = "cube_root"
    ABSOLUTE = "absolute"

    # Matrix
    MATRIX_MULTIPLICATION = "matrix_multiplication"
    MULTIPLY = "multiply"  # Elementwise multiplication
    DOT = "dot"
    TRANSPOSE = "transpose"
    MATRIX_SUM = "matrix_sum"
    SORT = "sort"
    SPLIT = "split"
    RESHAPE = "reshape"
    CONCATENATE = "concatenate"
    MIN = "min"
    MAX = "max"
    UNIQUE = "unique"
    ARGMAX = "argmax"
    ARGMIN = "argmin"
    EXPAND_DIMS = "expand_dims"
    INVERSE = "inv"
    GATHER = "gather"
    REVERSE = "reverse"
    STACK = "stack"
    TILE = "tile"
    SLICE = "slice"
    FIND_INDICES = "find_indices"

    # Comparison Operators
    GREATER = "greater"
    GREATER_EQUAL = "greater_equal"
    LESS = "less"
    LESS_EQUAL = "less_equal"
    EQUAL = "equal"
    NOT_EQUAL = "not_equal"

    # Logical
    LOGICAL_AND = "logical_and"
    LOGICAL_OR = "logical_or"
    LOGICAL_NOT = "logical_not"
    LOGICAL_XOR = "logical_xor"

    # Statistical
    MEAN = "mean"
    AVERAGE = "average"
    MODE = "mode"
    VARIANCE = "variance"
    MEDIAN = "median"
    STANDARD_DEVIATION = "standard_deviation"
    PERCENTILE = "percentile"
    RANDOM = "random"

    BINCOUNT = "bincount"
    WHERE = "where"
    SIGN = "sign"
    FOREACH = "foreach"


class TFJSOperators(Enum):
    SIGMOID = "sigmoid"
    SIN = "sin"
    SINH = "sinh"
    SOFTPLUS = "softplus"


class OpStatus(Enum):
    PENDING = "pending"
    COMPUTING = "computing"
    COMPUTED = "computed"
    FAILED = "failed"
