import pickle as pkl

import numpy as np
import ravop as R

from ravdl.v2 import NeuralNetwork
from ravdl.v2.layers import Dense, Dropout, BatchNormalization1D, Activation
from ravdl.v2.optimizers import Adam
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize


def to_categorical(x, n_col=None):
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot


def get_dataset():
    data = datasets.load_iris()
    X = data.data
    y = data.target
    X = normalize(X, axis=0)
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    y = to_categorical(y.astype("int"))
    n_samples, n_features = X.shape
    n_hidden = 15

    return X, X_test, y, y_test, n_hidden, n_features


def create_model(n_hidden, n_features):
    optimizer = Adam()
    model = NeuralNetwork(optimizer=optimizer, loss='SquareLoss')
    model.add(Dense(n_hidden, input_shape=(n_features,)))
    model.add(BatchNormalization1D())
    model.add(Dense(30))
    model.add(Dropout(0.9))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.summary()

    return model


def train(model, X, y, n_epochs=5, save_model=False):
    print('\nTraining...')
    model.fit(X, y, n_epochs=n_epochs, batch_size=25, save_model=save_model)
    pkl.dump(model, open("model.pkl", "wb"))
    return model


def test(model, X_test):
    print('\nTesting...')
    prediction = model.predict(X_test)
    prediction.persist_op(name='test_prediction')


def compile():
    R.activate()


def execute(participants=1):
    R.execute(participants=participants)
    R.track_progress()


def score(y_test):
    prediction = R.fetch_persisting_op(op_name="test_prediction")
    y_pred = np.argmax(prediction['result'].detach().numpy(), axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
