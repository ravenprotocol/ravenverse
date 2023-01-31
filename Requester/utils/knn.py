from dotenv import load_dotenv
load_dotenv()

import ravop as R
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def accuracy(y_true, y_pred):
    if not isinstance(y_true, R.Tensor):
        if not isinstance(y_true, R.Op):
            y_true = R.Tensor(y_true)
    if not isinstance(y_pred, R.Tensor):
        if not isinstance(y_pred, R.Op):
            y_pred = R.Tensor(y_pred)

    return R.div(R.sum(R.equal(y_pred, y_true)), y_pred.shape())

class KNNClassifier():
    def __init__(self, **kwargs):
        self._k = None
        self._n_c = None
        self._n = None
        self._X = None
        self._y = None
        self._labels = None

    def __euclidean_distance(self, X):
        X = R.expand_dims(X, axis=1)
        return R.square_root(R.sub(X, self._X).pow(R.t(2)).sum(axis=2))

    def fit(self, X, y, n_neighbours=None, n_classes=None):
        if n_neighbours is None or n_classes is None:
            raise Exception("Required params: n_neighbours, n_classes")

        self._X = R.t(X)
        self._y = R.t(y)
        self._k = n_neighbours
        self._n_c = n_classes
        self._n = len(X)

    def predict(self, X):
        n_q = len(X)
        X = R.Tensor(X)
        d_list = self.__euclidean_distance(X)
        print("calculating euclidian distance...")
        fe = d_list.foreach(operation='sort')
        sl = fe.foreach(operation='slice', begin=0, size=self._k)
        label = R.Tensor([])

        for i in range(n_q):
            row = d_list.gather(R.t([i])).squeeze()
            values = sl.gather(R.t([i])).squeeze()
            ind = row.find_indices(values).foreach(operation='slice', begin=0, size=1)
            y_neighbours = R.gather(self._y, ind.reshape(shape=[self._k]))
            label = label.concat(R.mode(y_neighbours))
        label.persist_op(name="test_prediction")

        self._labels = label


    def score(self, y_test):
        acc = accuracy(y_test, self._labels)
        acc.persist_op(name="score")


    @property
    def label(self):
        return self._label

    @property
    def points(self):
        return self._X

    def set_params(self, **kwargs):
        param_dict = {
            'labels': self._labels,
            'X': self._X,
            'y': self._y,
            'k': self.k
        }
        for i in kwargs.keys():
            if i in param_dict.keys():
                param_dict[i] = kwargs[i]
        return param_dict

    def get_params(self):
        param_dict = {
            'labels': self._labels,
            'X': self._X,
            'y': self._y,
            'k': self.k
        }
        return param_dict

def get_dataset():
    iris = load_iris()
    X = iris.data[:5500]
    y = iris.target[:5500]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.3)
    return X_train, X_test, y_train, y_test

def create_model():
    model = KNNClassifier()
    return model

def train(model, X_train, y_train, n_neighbours=20, n_classes=3):
    model.fit(X_train, y_train, n_neighbours=n_neighbours, n_classes=n_classes)
    return model

def test(model, X_test, y_test):
    model.predict(X_test)
    model.score(y_test=y_test)

def compile():
    R.activate()

def execute():
    R.execute()
    R.track_progress()

