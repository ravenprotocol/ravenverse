import ravop.core as R


def z_score(x, axis=None):
    if not isinstance(x, R.Tensor):
        x = R.Tensor(x)

    if axis is not None:
        mean = R.mean(x, axis=axis)
        std = R.std(x, axis=axis)
    else:
        mean = R.mean(x)
        std = R.std(x)

    return R.div(R.sub(x, mean), std)


def pearson_correlation(x, y):
    """
    Calculate linear correlation(pearson correlation)
    """

    if not isinstance(x, R.Tensor):
        x = R.Tensor(x)
    if not isinstance(y, R.Tensor):
        y = R.Tensor(y)

    a = R.sum(R.square(x))
    b = R.sum(R.square(y))

    n = a.output.shape[0]

    return R.div(R.sub(R.multiply(R.Scalar(n), R.sum(R.multiply(x, y))), R.multiply(R.sum(x), R.sum(y))),
                 R.multiply(R.square_root(R.sub(R.multiply(R.Scalar(n), a), R.square(b))),
                            R.square_root(R.sub(R.multiply(R.Scalar(n), b), R.square(b)))))


def normalize(x):
    """
    Normalize an array
    """
    if not isinstance(x, R.Tensor):
        x = R.Tensor(x)

    if len(x.output.shape) > 1:
        raise Exception("Unsupported input type")

    max = R.max(x)
    min = R.min(x)

    return R.div(R.sub(x, min), R.sub(max, min))


def standardize(x):
    """
    Standardize an array
    """
    if not isinstance(x, R.Tensor):
        x = R.Tensor(x)

    mean = R.mean(x)
    std = R.std(x)

    return R.div(R.sub(x, mean), std)
