from __future__ import print_function

import pickle as pkl

import numpy as np
import ravop as R


from ravdl.v2 import NeuralNetwork
from ravdl.v2.layers import Activation, Dense, BatchNormalization1D, BatchNormalization2D, Dropout, Conv2D, Flatten, MaxPooling2D
from ravdl.v2.optimizers import Adam, RMSprop

from sklearn import datasets
from sklearn.model_selection import train_test_split


def to_categorical(x, n_col=None):
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot


def get_dataset():
    data = datasets.load_digits()
    X = data.data
    y = data.target

    # Convert to one-hot encoding
    y = to_categorical(y.astype("int"))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

    # Reshape X to (n_samples, channels, height, width)
    X_train = X_train.reshape((-1, 1, 8, 8))
    X_test = X_test.reshape((-1, 1, 8, 8))

    return X_train, X_test, y_train, y_test


def create_model(X_test, y_test):
    optimizer = Adam()
    model = NeuralNetwork(optimizer=optimizer,
                          loss='CrossEntropy',
                          validation_data=(X_test, y_test))

    model.add(Conv2D(in_channels=1, out_channels=16, kernel_size=(3, 3), padding='same', input_shape=(1, 8, 8)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(kernel_size=(2, 2), stride=2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization2D(16))
    model.add(MaxPooling2D(kernel_size=(2, 2), stride=2))
    model.add(Conv2D(in_channels=16, out_channels=32, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(kernel_size=(2, 2), stride=2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization2D(32))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(BatchNormalization1D())
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.summary()
    return model


def train(model, X_train, y_train, n_epochs=10, persist_weights=False):
    model.fit(X_train, y_train, n_epochs=n_epochs, batch_size=256, persist_weights=persist_weights)
    pkl.dump(model, open("cnn_model.pkl", "wb"))
    return model

def test(model, X_test):
    print('\nTesting...')
    y_pred = model.test_on_batch(R.t(X_test))
    y_pred.persist_op(name='prediction')

def compile():
    R.activate()


def execute(participants=1):
    R.execute(participants=participants)
    R.track_progress()

def get_score(y_test):
    prediction = R.fetch_persisting_op(op_name="prediction")
    y_pred = np.argmax(prediction['result'].detach().numpy(), axis=-1)
    y_test = np.argmax(y_test, axis=-1)
    accuracy = np.sum(y_pred == y_test, axis=0) / len(y_test)
    return accuracy