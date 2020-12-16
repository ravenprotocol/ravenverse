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


def f1_score(y_true, y_pred, average):
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

    for i in sorted(set(y_true.output)):

      TP = R.sum(R.and(y_pred == i, y_true == i))
      TN = R.sum(R.and(y_pred =! i, y_true =! i))
      FP = R.sum(R.and(y_pred == i, y_true =! i))
      FN = R.sum(R.and(y_pred =! i, y_true == i))
      confusion.append([TP, TN, FP, FN])

    confusion = R.Tensor(confusion)

    if average=='macro':
      f = confusion.output
      for i in f:
        TP ,TN ,FP ,FN = i[0] ,i[1] ,i[2] ,i[3]
        Recall = R.div(TP, R.add(TP, FN))
        Precision = R.div(TP, R.add(TP, FP))

        if Precision == 0 or Recall == 0 or Recall == R.nan or Precision == R.nan:
          final.append(0)

        else:
          F1 = R.div(R.mul(R.Scalar(2), R.mul(Recall, Precision)),R.add(Recall ,Precision))
          final.append(F1)
        
      return R.mean(final)

    if average=='micro':

      TP = R.sum(confusion ,axis=0)[0]
      FP = R.sum(confusion ,axis=0)[2]
      FN = R.sum(confusion ,axis=0)[3]

      Recall = R.div(TP, R.add(TP, FN))
      Precision = R.div(TP, R.add(TP, FP))
      F1 = R.div(R.mul(R.Scalar(2), R.mul(Recall, Precision)),R.add(Recall ,Precision))
      return F1


  

def accuracy(y_true, y_pred):

  if not isinstance(y_true, R.Tensor):
      y_true = R.Tensor(y_true)
  if not isinstance(y_pred, R.Tensor):
      y_pred = R.Tensor(y_pred)

  acc = R.div(R.sum((y_pred == y_true)),y_pred.shape[0])
  return acc


def out_pred(y_true, y_pred, mode ,per_label=False):
  confusion = []

  if not isinstance(y_true, R.Tensor):
      y_true = R.Tensor(y_true)
  if not isinstance(y_pred, R.Tensor):
      y_pred = R.Tensor(y_pred)

  for i in sorted(set(y_true.output)):

    TP = R.sum(R.and(y_pred == i, y_true == i)) 
    TN = R.sum(R.and(y_pred =! i, y_true =! i))
    FP = R.sum(R.and(y_pred == i, y_true =! i))
    FN = R.sum(R.and(y_pred =! i, y_true == i))

    confusion.append([TP, TN, FP, FN])

  confusion = R.Tensor(confusion)

  if per_label:
    
    final = []
    f = confusion.output
    for i in f:

      TP ,TN ,FP ,FN = i[0] ,i[1] ,i[2] ,i[3]

      Precision = R.div(TP, R.add(TP, FP))
      Recall = R.div(TP, R.add(TP, FN))

      if mode == 'precision':

        if Precision == 0 or Precision == R.nan:
          final.append(0)

        else:

          final.append(Precision)

      if mode == 'recall':
        
        if Recall == 0 or Recall==R.nan:
          final.append(0)

        else:

          final.append(Recall)

    return final

  else:
    
    TP = R.sum(confusion ,axis=0)[0]
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

def aucroc(y_true, y_pred):
  confusion = []
  
  '''
  not completed
  '''
  
  if not isinstance(y_true, R.Tensor):
      y_true = R.Tensor(y_true)
  if not isinstance(y_pred, R.Tensor):
      y_pred = R.Tensor(y_pred)

  for i in sorted(set(y_true.output)):

    TP = R.sum(R.and(y_pred == i, y_true == i)) 
    TN = R.sum(R.and(y_pred =! i, y_true =! i))
    FP = R.sum(R.and(y_pred == i, y_true =! i))
    FN = R.sum(R.and(y_pred =! i, y_true == i))
    confusion.append([TP, TN, FP, FN])

  confusion = R.Tensor(confusion)
  
  TP = R.sum(confusion ,axis=0)[0]
  TN = R.sum(confusion ,axis=0)[1]
  FP = R.sum(confusion ,axis=0)[2]
  FN = R.sum(confusion ,axis=0)[3]

  tpr = R.div(TP, R.add(TP, FN))
  fpr = R.div(FP, R.add(TN, FP))
  return