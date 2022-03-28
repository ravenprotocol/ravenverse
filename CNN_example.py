from __future__ import print_function
from sklearn import datasets
import numpy as np

from ravdl.neural_networks import NeuralNetwork
from ravdl.neural_networks.layers import Conv2D, Dense, MaxPooling2D, Dropout, BatchNormalization, Activation, Flatten
from ravdl.neural_networks.optimizers import Adam
from ravdl.neural_networks.loss_functions import CrossEntropy

from sklearn.model_selection import train_test_split

import ravop as R

algo = R.Graph(name='cnn', algorithm='convolutional_neural_network', approach='distributed')

def to_categorical(x, n_col=None):
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot

optimizer = Adam()

data = datasets.load_digits()
X = data.data
y = data.target

# Convert to one-hot encoding
y = to_categorical(y.astype("int"))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# Reshape X to (n_samples, channels, height, width)
X_train = X_train.reshape((-1,1,8,8))
X_test = X_test.reshape((-1,1,8,8))

model = NeuralNetwork(optimizer=optimizer,
                    loss=CrossEntropy,
                    validation_data=(X_test, y_test))

model.add(Conv2D(n_filters=8, filter_shape=(4,4), stride=3, input_shape=(1,8,8), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Conv2D(n_filters=16, filter_shape=(4,4), stride=3, padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

train_err, val_err = model.fit(X_train, y_train, n_epochs=5, batch_size=256)

algo.end()