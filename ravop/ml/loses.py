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


def log_loss(y_true , y_pred , with_logit=True):
  if with_logit:
    y_pred = R.sigmoid(y_pred)
  else:
    pass
  y_pred = R.clip(y_pred, R.epsilon(), R.sub(R.Scalar(1), R.epsilon()))
  loss = R.elemul(R.Scalar(-1) ,R.mean(R.elemul(y_true ,R.natlog(y_pred)) ,R.elemul((R.sub(R.Scalar(1), y_true)) ,R.natlog(R.sub(R.Scalar(1),y_pred)))))

  return loss


def one_hot_cross_entropy(y_true, y_pred, with_logit=True):
  if with_logit:
    y_pred = R.softmax(y_pred)
  else:
    pass
  y_pred = R.clip(y_pred, R.epsilon(), R.div(R.Scalar(1) ,R.epsilon()))
  N = y_pred.shape[0]
  loss = R.div(R.elemul(R.Scalar(-1),R.mul(R.sum(y_true ,R.natlog(R.add(y_pred,1e-9))))),R.Scalar(N))
  return loss

def sparse_cross_entropy(y_true, y_pred, with_logit=True):
  if with_logit:
    y_pred = R.softmax(y_pred)
  else:
    pass
  y_pred = R.clip(y_pred, R.epsilon(), R.div(R.Scalar(1) ,R.epsilon()))
  N = y_pred.shape[0]
  loss = R.elemul(R.Scalar(-1),R.div(R.sum(R.natlog(y_pred[R.len(y_pred),y_true])),R.Scalar(N)))
  return loss
