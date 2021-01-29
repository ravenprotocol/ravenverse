import ravop.core as R


def z_score(x, axis=None):
    if not isinstance(x, R.Tensor):
        x = R.Tensor(x)

    mean = R.mean(x, axis=axis)
    std = R.std(x, axis=axis)
    return R.div(R.sub(x, mean), std)
