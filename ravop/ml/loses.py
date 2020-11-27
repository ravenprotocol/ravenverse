import ravop.core as R

def mean_ab_error(y_true ,y_pred):

  if not isinstance(y_true, R.Tensor):
        y_true = R.Tensor(y_true)
  if not isinstance(y_pred, R.Tensor):
        y_pred = R.Tensor(y_pred)

  return R.mean(R.sqrt(R.square(R.sub(y_true, y_pred))))

def mean_sq_error(y_true ,y_pred):

  if not isinstance(y_true, R.Tensor):
        y_true = R.Tensor(y_true)
  if not isinstance(y_pred, R.Tensor):
        y_pred = R.Tensor(y_pred)

  return R.mean(R.sqr(R.sub(y_true, y_pred)))

def root_mean_sq_error(y_true ,y_pred):

  if not isinstance(y_true, R.Tensor):
        y_true = R.Tensor(y_true)
  if not isinstance(y_pred, R.Tensor):
        y_pred = R.Tensor(y_pred)

  return R.sqrt(mean_sq_error(y_true, y_pred))
