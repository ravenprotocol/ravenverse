import ravop.core as R
import numpy as np
from ravop.ml.activations import sigmoid, softmax


def softmax_num(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def sigmoid_num(x):
    return 1 / (1 + np.exp(-x))


def mean_absolute_error(y_true, y_pred):
    """
    Mean Absolute Error

    Usage:
            y_true = np.array([[0., 1.], [0., 0.]])
            y_pred = np.array([[1., 1.], [0., 0.]])

            loss = mean_absolute_error(y_true , y_pred)

    """
    while True:

        if y_pred.status == 'computed' and y_true.status == 'computed':
            break

        else:
            pass

    if not isinstance(y_true, R.Tensor):
        y_true = R.Tensor(y_true)
    if not isinstance(y_pred, R.Tensor):
        y_pred = R.Tensor(y_pred)

    return R.mean(R.abs(R.sub(y_pred, y_true)))


def mean_squared_error(y_true, y_pred):
    """
    Mean Squared Error

    Usage:
            y_true = np.array([[0., 1.], [0., 0.]])
            y_pred = np.array([[1., 1.], [0., 0.]])
            loss = mean_squared_error(y_true , y_pred)

    """
    while True:

        if y_pred.status == 'computed' and y_true.status == 'computed':
            break

        else:
            print('wait until y_true and pred get computed')
            pass

    if not isinstance(y_true, R.Tensor):
        y_true = R.Tensor(y_true)
    if not isinstance(y_pred, R.Tensor):
        y_pred = R.Tensor(y_pred)

    return R.mean(R.pow(R.sub(y_true, y_pred), R.Scalar(2)))


def root_mean_squared_error(y_true, y_pred):
    """
    Root Mean Squared Error

    Usage:
            y_true = np.array([[0., 1.], [0., 0.]])
            y_pred = np.array([[1., 1.], [0., 0.]])

            loss = root_mean_squared_error(y_true , y_pred)

    """
    if not isinstance(y_true, R.Tensor):
        y_true = R.Tensor(y_true)
    if not isinstance(y_pred, R.Tensor):
        y_pred = R.Tensor(y_pred)

    while True:

        if y_pred.status == 'computed' and y_true.status == 'computed':
            break

        else:
            print('wait until y_true and pred get computed')
            pass

    return R.pow(mean_squared_error(y_true, y_pred), R.Scalar(0.5))


def mean_squared_log_error(y_true, y_pred):
    """
    Mean Squared Log Error

    Usage:
            y_true = np.array([[0., 1.], [0., 0.]])
            y_pred = np.array([[1., 1.], [0., 0.]])
            loss = mean_squared_log_error(y_true , y_pred)
    """
    if not isinstance(y_true, R.Tensor):
        y_true = R.Tensor(y_true)
    if not isinstance(y_pred, R.Tensor):
        y_pred = R.Tensor(y_pred)

    while True:

        if y_pred.status == 'computed' and y_true.status == 'computed':
            break

        else:
            print('wait until y_true and pred get computed')
            pass

    return R.mean(R.pow(R.sub(R.natlog(R.add(y_true, R.one())), R.natlog(R.add(y_pred, R.one()))), R.Scalar(2)))


def log_loss(y_true, y_pred, with_logit=True):
    """
        Compute log loss if you want to perform it with sigmoid then use the Flag of with_logit=True

    Usage:
        y_true = np.array([[0, 1], [0, 0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])
        loss = log_loss(y_true , y_pred)

    """

    if not isinstance(y_true, R.Tensor):
        y_true = R.Tensor(y_true)
    if not isinstance(y_pred, R.Tensor):
        y_pred = R.Tensor(y_pred)

    while True:

        if y_pred.status == 'computed' and y_true.status == 'computed':
            break

        else:
            print('wait until y_true and pred get computed')
            pass

    if with_logit:
        y_pred = sigmoid(y_pred)

    else:
        pass

    # y_pred = R.clip(y_pred, R.epsilon(), R.sub(R.Scalar(1), R.epsilon()))
    loss = R.mul(R.Scalar(-1), R.mean(R.add(R.mul(y_true, R.natlog(y_pred)),
                                            R.mul(R.sub(R.Scalar(1), y_true), R.natlog(R.sub(R.Scalar(1), y_pred))))))

    return loss


def one_hot_cross_entropy(y_true, y_pred, with_logit=False):
    """
        Compute one_hot_cross_entropy if you want to perform it with softmax then use the Flag of with_logit=True

    Usage:
        y_true = np.array([[0, 1, 0], [0, 0, 1])
        y_pred = np.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
        loss = one_hot_cross_entropy(y_true , y_pred)

    """

    if not isinstance(y_true, R.Tensor):
        y_true = R.Tensor(y_true)
    if not isinstance(y_pred, R.Tensor):
        y_pred = R.Tensor(y_pred)

    while True:

        if y_pred.status == 'computed' and y_true.status == 'computed':
            n = y_pred.output.shape[0]
            break

        else:
            print('wait until y_true and pred get computed')
            pass

    if with_logit:
        y_pred = softmax(y_pred)
    else:
        pass

    # y_pred = R.clip(y_pred, R.epsilon(), R.div(R.Scalar(1), R.epsilon()))

    y_true = R.Tensor(y_true)
    y_pred = R.Tensor(y_pred)

    loss = R.div(R.mul(R.Scalar(-1), R.sum(R.mul(y_true, R.natlog(y_pred)))), R.Scalar(n))
    return loss


def sparse_cross_entropy(y_true, y_pred, with_logit=False):
    """
        Compute sparse_cross_entropy if you want to perform it with softmax then use the Flag of with_logit=True

    Usage:
        y_true = np.array([[0, 1, 0], [0, 0, 1])
        y_pred = np.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
        loss = sparse_cross_entropy(y_true , y_pred)

    """

    if not isinstance(y_true, R.Tensor):
        y_true = R.Tensor(y_true)
    if not isinstance(y_pred, R.Tensor):
        y_pred = R.Tensor(y_pred)

    if with_logit:
        y_pred = softmax(y_pred)

    else:
        pass

    while True:

        if y_pred.status == 'computed' and y_true.status == 'computed':

            a = y_pred.output
            n = y_pred.output.shape[0]
            f = a[range(len(a)), y_true.output]
            break

        else:
            print('wait until y_true and pred get computed')
            pass

    # y_pred = R.clip(y_pred, R.epsilon(), R.div(R.Scalar(1), R.epsilon()))
    f = R.Tensor(f)
    loss = R.mul(R.Scalar(-1), R.div(R.sum(R.natlog(f)), R.Scalar(n)))

    return loss


def categorical_hinge(y_true, y_pred):
    """
        Compute categorical_hinge

    Usage:
        y_true = np.array([[0, 1], [0, 0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])
        loss = categorical_hinge(y_true , y_pred)

    """

    if not isinstance(y_true, R.Tensor):
        y_true = R.Tensor(y_true)
    if not isinstance(y_pred, R.Tensor):
        y_pred = R.Tensor(y_pred)

    while True:

        if y_pred.status == 'computed' and y_true.status == 'computed':
            break

        else:
            print('wait until y_true and pred get computed')
            pass

    neg = R.max(R.mul(R.sub(R.Scalar(1), y_true), y_pred))
    pos = R.sum(R.mul(y_true, y_pred))
    loss = R.max(R.Tensor([R.add(R.sub(neg, pos), R.Scalar(1)), R.Scalar(0)]))
    return loss


def huber(y_true, y_pred, d=1.0):
    """
        Compute huber regression loss
        delta or d:	A float, the point where the Huber loss function changes from a quadratic to linear.

    Usage:
        y_true = np.array([[0, 1], [0, 0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])
        loss = huber(y_true , y_pred)

    """

    while True:

        if y_pred.status == 'computed' and y_true.status == 'computed':
            break

        else:
            print('wait until y_true and pred get computed')
            pass

    d = R.Scalar(d)
    x = R.sub(y_true, y_pred)

    if R.abs(x).less_equal(d):
        return R.sum(R.mul(d, R.square(x)))

    if R.abs(x).greater(d):
        return R.sum(R.add(R.square(d), R.mul(d, R.sub(R.abs(x), d))))


def kl_div_loss(y_true, y_pred):
    """
        Compute kl_divergence loss

    Usage:
        y_true = np.array([[0, 1], [0, 0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])
        loss = kl_div_loss(y_true , y_pred)
    """

    if not isinstance(y_true, R.Tensor):
        y_true = R.Tensor(y_true)
    if not isinstance(y_pred, R.Tensor):
        y_pred = R.Tensor(y_pred)

    while True:

        if y_pred.status == 'computed' and y_true.status == 'computed':
            break

        else:
            print('wait until y_true and pred get computed')
            pass

    # y_pred = R.clip(y_pred, R.epsilon(), R.sub(R.Scalar(1), R.epsilon()))

    return R.sum(R.mul(y_true, R.natlog(R.div(y_true, y_pred))))


def poisson_loss(y_true, y_pred):
    """
        Compute poisson_loss

        Usage:
            y_true = np.array([[0., 1.], [0., 0.]])
            y_pred = np.array([[1., 1.], [0., 0.]])

            loss = poisson_loss(y_true , y_pred)
    """

    if not isinstance(y_true, R.Tensor):
        y_true = R.Tensor(y_true)
    if not isinstance(y_pred, R.Tensor):
        y_pred = R.Tensor(y_pred)

    while True:

        if y_pred.status == 'computed' and y_true.status == 'computed':
            break

        else:
            print('wait until y_true and pred get computed')
            pass

    # y_pred = R.clip(y_pred, R.epsilon(), R.sub(R.Scalar(1) ,R.epsilon()))

    return R.sum(R.sub(y_pred, R.mul(y_true, R.natlog(y_pred))))
