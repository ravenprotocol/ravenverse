import os

from dotenv import load_dotenv

load_dotenv()

from sklearn import datasets
import numpy as np
import pickle as pkl

from ravdl.neural_networks import NeuralNetwork
from ravdl.neural_networks.layers import Dense, Dropout, BatchNormalization, Activation
from ravdl.neural_networks.optimizers import RMSprop
from ravdl.neural_networks.loss_functions import SquareLoss

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

import ravop as R

R.initialize(ravenverse_token=os.environ.get("TOKEN"))
R.flush()
R.Graph(name='ann', algorithm='neural_network', approach='distributed')

data = datasets.load_iris()
X = data.data
y = data.target
X = normalize(X, axis=0)
X, X_test, y, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


def to_categorical(x, n_col=None):
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot


y = to_categorical(y.astype("int"))
n_samples, n_features = X.shape
n_hidden = 15

optimizer = RMSprop()

model = NeuralNetwork(optimizer=optimizer, loss=SquareLoss)
model.add(Dense(n_hidden, input_shape=(n_features,)))
model.add(BatchNormalization())
model.add(Dense(30))
model.add(Dropout(0.9))
model.add(Dense(3))
model.add(Activation('softmax'))

model.summary()

print('\nTraining...')

model.fit(X, y, n_epochs=5, batch_size=25, save_model=True)
pkl.dump(model, open("model.pkl", "wb"))

print('\nTesting...')

prediction = model.predict(X_test)
prediction.persist_op(name='test_prediction')

R.activate()
