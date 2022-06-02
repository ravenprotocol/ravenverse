
from sklearn import datasets
import numpy as np

from ravdl.neural_networks import NeuralNetwork
from ravdl.neural_networks.layers import Dense, Dropout, BatchNormalization, Activation
from ravdl.neural_networks.optimizers import RMSprop
from ravdl.neural_networks.loss_functions import SquareLoss

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import ravop.ravop as R

R.initialize(ravenverse_token='YOUR_TOKEN')
algo = R.Graph(name='ann', algorithm='neural_network', approach='distributed')

data = datasets.load_iris()
X = data.data
y = data.target
X=normalize(X,axis=0)
X, X_test, y , y_test = train_test_split(X, y, test_size=0.33)

def to_categorical(x, n_col=None):
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot

y = to_categorical(y.astype("int"))
n_samples, n_features = X.shape
n_hidden = 15
print("No of samples:",n_samples)

optimizer = RMSprop()

model = NeuralNetwork(optimizer=optimizer,loss=SquareLoss)
model.add(Dense(n_hidden, input_shape=(n_features,)))
model.add(BatchNormalization())
model.add(Dense(30))
model.add(Dropout(0.9))
model.add(Dense(3))
model.add(Activation('softmax'))

model.summary()

print('\nTraining...')

train_err = model.fit(X, y, n_epochs=1, batch_size=25)

print('\nTesting...')

y_pred = np.argmax(model.predict(X_test)(),axis=1)

accuracy = accuracy_score(y_test, y_pred)

print ("Accuracy:", accuracy)

algo.end()