from ravop.core import Tensor, Scalar


def f1_score(y_true, y_pred):
    pass


def mean_squared_error(y_true, y_pred):
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true)
    if not isinstance(y_pred, Tensor):
        y_pred = Tensor(y_pred)

    a = Tensor(y_true).sub(y_pred)
    b = a.square()
    c = b.sum()
    d = Scalar(1).div(y_pred.shape[0]).elemul(c) 
    return d


def root_mean_squared_error(y_true, y_pred):
    pass


def mean_absolute_error(y_true, y_pred):
    pass
