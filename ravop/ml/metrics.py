import ravop.core as R


def r2_score(y_true, y_pred):

  if not isinstance(y_true, R.Tensor):
      y_true = R.Tensor(y_true)
  if not isinstance(y_pred, R.Tensor):
      y_pred = R.Tensor(y_pred)    
  
  scalar1 = R.Scalar(1)    
        
  SS_res = R.sum(R.square(R.sub(y_true, y_pred)))
  SS_tot = R.sum(R.square(R.sub(y_true, R.mean(y_true))))  

  return R.sub(scalar1, R.div(SS_res, R.add(SS_tot, R.epsilon())))


def f1_score(true_labels, pred_labels, average):
    """
    average argument:
    
        micro( Find F1 for each label seperately and then average all
                of the F1 scores also applicable in binary case )
                
        macro( Calculate the TP, TN, FP, FN Globally and then calculate
                the F1 scores )
    """
  confusion = []
  final = []

  if not isinstance(y_true, R.Tensor):
      y_true = R.Tensor(y_true)
  if not isinstance(y_pred, R.Tensor):
      y_pred = R.Tensor(y_pred)  

  for i in R.sort(set(true_labels)):

    TP = R.sum(R.and(pred_labels == i, true_labels == i)) 
    TN = R.sum(R.and(pred_labels =! i, true_labels =! i))
    FP = R.sum(R.and(pred_labels == i, true_labels =! i))
    FN = R.sum(R.and(pred_labels =! i, true_labels == i))

    confusion.append([TP, TN, FP, FN])

  confusion = R.Tensor(confusion)

  if average=='macro':

    for i in confusion:
      TP ,TN ,FP ,FN = i[0] ,i[1] ,i[2] ,i[3]
      Recall = R.div(TP, R.add(TP, FN))
      Precision = R.div(TP, R.add(TP, FP))

      if Precision == 0 or Recall == 0 
          or Recall == np.nan or Precision == np.nan:
        final.append(0)

      else:
        F1 = R.div(R.elemul(R.Scalar(2), R.elemul(Recall, Precision)),R.sum(Recall ,Precision))
        final.append(F1)
        
    return R.mean(final)

  if average=='micro':

    confusion = R.Tensor(confusion)
    TP = R.sum(confusion ,axis=0)[0]
    TN = R.sum(confusion ,axis=0)[1]
    FP = R.sum(confusion ,axis=0)[2]
    FN = R.sum(confusion ,axis=0)[3]

    Recall = R.div(TP, R.add(TP, FN))
    Precision = R.div(TP, R.add(TP, FP))
    F1 = R.div(R.elemul(R.Scalar(2), R.elemul(Recall, Precision)),R.sum(Recall ,Precision))
    return F1


  

def accuracy(y_true, y_pred):

  if not isinstance(y_true, R.Tensor):
      y_true = R.Tensor(y_true)
  if not isinstance(y_pred, R.Tensor):
      y_pred = R.Tensor(y_pred)

  y_pred = np.array([ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
  y_val = np.array([0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
  accuracy = R.div(R.sum((y_pred == y_val)),y_pred.shape[0])
  return accuracy


def out_pred(y_true, y_pred, per_label=False, mode):

  if not isinstance(y_true, R.Tensor):
      y_true = R.Tensor(y_true)
  if not isinstance(y_pred, R.Tensor):
      y_pred = R.Tensor(y_pred)

  for i in sorted(set(y_true)):

    TP = R.sum(R.and(y_pred == i, y_true == i)) 
    TN = R.sum(R.and(y_pred =! i, y_true =! i))
    FP = R.sum(R.and(y_pred == i, y_true =! i))
    FN = R.sum(R.and(y_pred =! i, y_true == i))

    confusion.append([TP, TN, FP, FN])

  confusion = R.Tensor(confusion)

  if per_label:
    
    final = []

    for i in confusion:
      TP ,TN ,FP ,FN = i[0] ,i[1] ,i[2] ,i[3]

      Precision = R.div(TP, R.add(TP, FP))
      Recall = R.div(TP, R.add(TP, FN))

      if mode == 'precision':

        if Precision == 0 or Precision == np.nan:
          final.append(0)

        else:

          final.append(Precision)

      if mode == 'recall':
        
        if Recall == 0 or Recall==np.nan:
          final.append(0)

        else:

          final.append(Recall)

    return final

  else:

    confusion = R.Tensor(confusion)
    TP = R.sum(confusion ,axis=0)[0]
    TN = R.sum(confusion ,axis=0)[1]
    FP = R.sum(confusion ,axis=0)[2]
    FN = R.sum(confusion ,axis=0)[3]

    if mode == 'precision':

      Precision = R.div(TP, R.add(TP, FP))
      return Precision

    if mode == 'recall':

      Recall = R.div(TP, R.add(TP, FN))
      return Recall

def recall(y_true, y_pred, per_label):

  return out_pred(y_true, y_pred, per_label=per_label,mode='recall')

def precision(y_true, y_pred, per_label):

  return out_pred(y_true, y_pred, per_label=per_label,mode='precision')

def AUCROC(y_true, y_pred):

  '''
  not completed

  '''

  for i in sorted(set(y_true)):

    TP = R.sum(R.and(y_pred == i, y_true == i)) 
    TN = R.sum(R.and(y_pred =! i, y_true =! i))
    FP = R.sum(R.and(y_pred == i, y_true =! i))
    FN = R.sum(R.and(y_pred =! i, y_true == i))

    confusion.append([TP, TN, FP, FN])

  confusion = R.Tensor(confusion)

  tpr = R.div(TP, R.add(TP, FN))
  fpr = R.div(FP, R.add(TN, FP)) 
