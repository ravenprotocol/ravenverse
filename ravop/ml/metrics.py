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

    while True:

        if y_pred.status == 'computed' and y_true.status == 'computed':
            break

        else:
            pass

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

    while True:

        if y_pred.status == 'computed':
            print('y_pred_computed')
            break

        else:
            pass

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

            while True:

                if Recall.status == 'computed' and Precision.status == 'computed':
                    print('precision recall computed')
                    p = Precision.output
                    r = Recall.output
                    break

                else:
                    pass

            # if Precision.equal(R.Scalar(0)) or Precision.equal(R.Scalar(np.nan)) or Recall.equal(R.Scalar(0)) or
            # \Recall.equal(R.Scalar(np.nan)):

            if p == 0 or p == np.nan or r == 0 or r == np.nan:
                final.append(R.Tensor(0))

            else:

                F1 = R.div(R.mul(R.Scalar(2), R.mul(Recall, Precision)), R.add(Recall, Precision))
                final.append(F1)

        a = R.div(R.sum(R.Tensor(final)), R.Scalar(len(confusion)), name='f1')

        while True:

            if a.status == 'computed':
                print('f1 computed')
                return a
            else:
                print('f1 computing')
                pass

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

        while True:

            if F1.status == 'computed':
                print('f1 computed')
                return F1
            else:
                print('f1 computing')
                pass


def accuracy(y_true, y_pred):
    if not isinstance(y_true, R.Tensor):
        y_true = R.Tensor(y_true)
    if not isinstance(y_pred, R.Tensor):
        y_pred = R.Tensor(y_pred)

    while True:

        if y_pred.status == 'computed' and y_true.status == 'computed':
            n = y_pred.output.shape[0]
            break

        else:
            pass

    acc = R.div(R.sum((y_pred.equal(y_true))), R.Scalar(n))
    return acc


def out_pred(y_true, y_pred, mode, per_label=False):
    """
    helper function for precision and recall
    """
    confusion = []

    while True:

        if y_pred.status == 'computed' and y_true.status == 'computed':
            break

        else:
            pass

    if not isinstance(y_true, R.Tensor):
        y_true = R.Tensor(y_true)
    if not isinstance(y_pred, R.Tensor):
        y_pred = R.Tensor(y_pred)

    while True:

        if y_pred.status == 'computed':
            print('y_pred_computed')
            break

        else:
            pass

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

                while True:

                    if Precision.status == 'computed':
                        print('precision computed')
                        p = Precision.output
                        break

                    else:
                        pass

                # if Precision.equal(R.Scalar(0)) or Precision.equal(R.Scalar(np.nan)) or Recall.equal(R.Scalar(0)) or
                # \Recall.equal(R.Scalar(np.nan)):

                if p == 0 or p == np.nan:
                    final.append(R.Tensor(0))

                else:
                    final.append(Precision)

            if mode == 'recall':

                while True:

                    if Recall.status == 'computed':
                        print('precision computed')
                        r = Recall.output
                        break

                    else:
                        pass

                # if Precision.equal(R.Scalar(0)) or Precision.equal(R.Scalar(np.nan)) or Recall.equal(R.Scalar(0)) or
                # \Recall.equal(R.Scalar(np.nan)):

                if r == 0 or r == np.nan:
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
    if not isinstance(y_true, R.Tensor):
        y_true = R.Tensor(y_true)
    if not isinstance(y_pred, R.Tensor):
        y_pred = R.Tensor(y_pred)

    l = []
    """
    
    Usage:
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0.05, 0.95, 0.0])
        loss = aucroc(y_true , y_pred)
    """

    while True:

        if y_pred.status == 'computed' and y_true.status == 'computed':
            break

        else:
            pass

    partitions = 100

    for j in range(partitions + 1):
        y_pred = R.mul(y_pred.greater_equal(R.div(R.Tensor(j), R.Tensor(partitions))), R.Tensor(1))

        confusion = []

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
        l.append([tpr, fpr])
        
    rectangle_roc = R.Tensor(0)
    for k in range(partitions):
        rectangle_roc = R.add(rectangle_roc, R.mul(R.sub(l[k][1], l[k + 1][1]), l[k][0]))
    return rectangle_roc
