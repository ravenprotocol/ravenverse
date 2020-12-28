import ravop.core as R
import numpy as np


def r2_score(y_true, y_pred):
    """

    :param y_true: numpy.ndarray
    :param y_pred: numpy.ndarray
    :return: r2_score
    """

    if not isinstance(y_true, R.Tensor):
        y_true = R.Tensor(y_true)
    if not isinstance(y_pred, R.Tensor):
        y_pred = R.Tensor(y_pred)

    scalar1 = R.Scalar(1)

    SS_res = R.sum(R.square(R.sub(y_true, y_pred)))
    SS_tot = R.sum(R.square(R.sub(y_true, R.mean(y_true))))

    return R.sub(scalar1, R.div(SS_res, R.add(SS_tot, R.epsilon())))


def f1_score(y_true, y_pred, average='macro'):
    """
    average argument:

        micro( Find F1 for each label seperately and then average all
                of the F1 scores also applicable in binary case )

        macro( Calculate the TP, TN, FP, FN Globally and then calculate
                the F1 scores )

        return F1 score (tensor)
    """

    confusion = []
    final = []

    if not isinstance(y_true, R.Tensor):
        y_true = R.Tensor(y_true)
    if not isinstance(y_pred, R.Tensor):
        y_pred = R.Tensor(y_pred)

    if y_true.status == 'computed':
        a = y_true.output

        for i in sorted(set(a)):

            i = R.Tensor(i)
            TP = R.sum(R.logical_and(y_pred.equal(i), y_true.equal(i)))
            TN = R.sum(R.logical_and(y_pred.not_equal(i), y_true.not_equal(i)))
            FP = R.sum(R.logical_and(y_pred.equal(i), y_true.not_equal(i)))
            FN = R.sum(R.logical_and(y_pred.not_equal(i), y_true.equal(i)))
            confusion.append([TP, TN, FP, FN])

    if average == 'macro':

        for i in confusion:
            TP, TN, FP, FN = i[0], i[1], i[2], i[3]

            Recall = R.div(TP, R.add(TP, FN))
            Precision = R.div(TP, R.add(TP, FP))

            # if Recall.status == 'pending' and Precision.status == 'pending':
            #    p = Precision.output
            #    r = Recall.output

            if Precision.equal(R.Tensor(0)) or Precision.equal(R.Tensor(np.nan)) or Recall.equal(R.Tensor(0)) or \
                    Recall.equal(R.Tensor(np.nan)):
                final.append(R.Tensor(0))

            else:
                F1 = R.div(R.mul(R.Scalar(2), R.mul(Recall, Precision)), R.add(Recall, Precision))
                final.append(F1)

        return R.div(R.Tensor(final), R.Scalar(len(confusion)))

    if average == 'micro':

        TP = R.Scalar(0)
        FP = R.Scalar(0)
        FN = R.Scalar(0)

        for i in confusion:
            TP = R.add(TP, i[0])
            FP = R.add(FP, i[2])
            FN = R.add(FN, i[3])

        Recall = R.div(TP, R.add(TP, FN))
        Precision = R.div(TP, R.add(TP, FP))
        F1 = R.div(R.mul(R.Scalar(2), R.mul(Recall, Precision)), R.add(Recall, Precision))

        return F1


def accuracy(y_true, y_pred):
    n = y_pred.shape[0]
    if not isinstance(y_true, R.Tensor):
        y_true = R.Tensor(y_true)
    if not isinstance(y_pred, R.Tensor):
        y_pred = R.Tensor(y_pred)

    acc = R.div(R.sum((y_pred.equal(y_true))), R.Scalar(n))
    return acc


def out_pred(y_true, y_pred, mode, per_label=False):

    """
    helper function for precision and recall
    """
    confusion = []

    if not isinstance(y_true, R.Tensor):
        y_true = R.Tensor(y_true)
    if not isinstance(y_pred, R.Tensor):
        y_pred = R.Tensor(y_pred)

    if y_true.status == 'computed':
        a = y_true.output

        for i in sorted(set(a)):

            i = R.Tensor(i)
            TP = R.sum(R.logical_and(y_pred.equal(i), y_true.equal(i)))
            TN = R.sum(R.logical_and(y_pred.not_equal(i), y_true.not_equal(i)))
            FP = R.sum(R.logical_and(y_pred.equal(i), y_true.not_equal(i)))
            FN = R.sum(R.logical_and(y_pred.not_equal(i), y_true.equal(i)))
            confusion.append([TP, TN, FP, FN])

    if per_label:

        final = []
        for i in confusion:

            TP, TN, FP, FN = i[0], i[1], i[2], i[3]
            Precision = R.div(TP, R.add(TP, FP))
            Recall = R.div(TP, R.add(TP, FN))

            if mode == 'precision':

                if Precision.equal(R.Tensor(0)) or Precision.equal(R.Tensor(np.nan)):
                    final.append(R.Tensor(0))

                else:
                    final.append(Precision)

            if mode == 'recall':

                if Recall.equal(R.Tensor(0)) or Recall.equal(R.Tensor(np.nan)):
                    final.append(R.Tensor(0))
                  
                else:
                    final.append(Recall)
                    
    else:

        TP = R.Scalar(0)
        FP = R.Scalar(0)
        FN = R.Scalar(0)

        for i in confusion:
            TP = R.add(TP, i[0])
            FP = R.add(FP, i[2])
            FN = R.add(FN, i[3])

        if mode == 'precision':
            Precision = R.div(TP, R.add(TP, FP))
            return Precision

        if mode == 'recall':
            Recall = R.div(TP, R.add(TP, FN))
            return Recall


def recall(y_true, y_pred, per_label=True):
    """

    :param y_true: numpy.ndarray
    :param y_pred: numpy.ndarray
    :param per_label: True if macro else micro
    :return: score
    """

    return out_pred(y_true, y_pred, per_label=per_label, mode='recall')


def precision(y_true, y_pred, per_label=True):
    """

    :param y_true: numpy.ndarray
    :param y_pred: numpy.ndarray
    :param per_label: True if macro else micro
    :return: score
    """
    return out_pred(y_true, y_pred, per_label=per_label, mode='precision')


def aucroc(y_true, y_pred):
    """

    :param y_true: numpy.ndarray
    :param y_pred: numpy.ndarray
    :return: incomplete
    """

    confusion = []

    if not isinstance(y_true, R.Tensor):
        y_true = R.Tensor(y_true)
    if not isinstance(y_pred, R.Tensor):
        y_pred = R.Tensor(y_pred)

    if y_true.status == 'computed':
        a = y_true.output

        for i in sorted(set(a)):

            i = R.Tensor(i)
            TP = R.sum(R.logical_and(y_pred.equal(i), y_true.equal(i)))
            TN = R.sum(R.logical_and(y_pred.not_equal(i), y_true.not_equal(i)))
            FP = R.sum(R.logical_and(y_pred.equal(i), y_true.not_equal(i)))
            FN = R.sum(R.logical_and(y_pred.not_equal(i), y_true.equal(i)))
            confusion.append([TP, TN, FP, FN])

    TP = R.Scalar(0)
    TN = R.Scalar(0)
    FP = R.Scalar(0)
    FN = R.Scalar(0)

    for i in confusion:
        TP = R.add(TP, i[0])
        TN = R.add(FN, i[1])
        FP = R.add(FP, i[2])
        FN = R.add(FN, i[3])

    tpr = R.div(TP, R.add(TP, FN))
    fpr = R.div(FP, R.add(TN, FP))
    return
