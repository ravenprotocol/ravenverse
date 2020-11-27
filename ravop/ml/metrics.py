import ravop.core as R


def r2_score(y_true, y_pred):
    """
    Calculate r2 score
    """
    # Convert to RavOp tensor if in ndarray
    if not isinstance(y_true, R.Tensor):
        y_true = R.Tensor(y_true)
    if not isinstance(y_pred, R.Tensor):
        y_pred = R.Tensor(y_pred)

    scalar1 = R.Scalar(1)

    SS_res = R.sum(R.square(R.sub(y_true, y_pred)))
    SS_tot = R.sum(R.square(R.sub(y_true, R.mean(y_true))))

    return R.sub(scalar1, R.div(SS_res, R.add(SS_tot, R.epsilon())))

