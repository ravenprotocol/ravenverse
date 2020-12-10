import ravop.core as R


def softmax(x):
    """
    Softmax Activation Function
    """
    exp = R.exp(x)
    return R.div(exp, R.sum(exp))


def sigmoid(x):
    """
    Sigmoid Activation Function
    """
    return R.div(R.one(), R.add(R.one(), R.exp(R.multiply(R.minus_one(), x))))


def tanh(x):
    """
    Tanh Activation Function
    """
    return R.div(R.sub(R.exp(x), R.exp(R.mul(R.minus_one(), x))), R.add(R.exp(x), R.exp(R.mul(R.minus_one(), x))))