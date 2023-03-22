from dotenv import load_dotenv
load_dotenv()

import numpy as np
import os
import ravop as R

from ravdl.v2 import Functional
from ravdl.v2.layers import Dense, Dropout, BatchNormalization1D, Activation, CustomLayer
from ravdl.v2.optimizers import Adam
from ravdl.v2.utils.data_manipulation import batch_iterator
from sklearn.metrics import accuracy_score
from utils import ann

# Initialize and create graph
R.initialize(ravenverse_token=os.environ.get("TOKEN"))
R.flush()
R.Graph(name = 'ann', algorithm = 'neural_network', approach = 'distributed')

X, X_test, y, y_test, n_hidden, n_features = ann.get_dataset()

class CustomLayer1(CustomLayer):
    def __init__(self) -> None:
        super().__init__()
        self.d1 = Dense(n_hidden, input_shape=(n_features,))
        self.bn1 = BatchNormalization1D(momentum=0.99, epsilon=0.01)

    def _forward_pass_call(self, input, training=True):
        o = self.d1._forward_pass(input)
        o = self.bn1._forward_pass(o, training=training)
        return o

class CustomLayer2(CustomLayer):
    def __init__(self) -> None:
        super().__init__()
        self.d1 = Dense(30)
        self.dropout = Dropout(0.9)
        self.d2 = Dense(3)

    def _forward_pass_call(self, input, training=True):
        o = self.d1._forward_pass(input)
        o = self.dropout._forward_pass(o, training=training)
        o = self.d2._forward_pass(o)
        return o

class ANNModel(Functional):
    def __init__(self, optimizer):
        super().__init__()
        self.custom_layer1 = CustomLayer1()
        self.custom_layer2 = CustomLayer2()
        self.act = Activation('softmax')
        self.initialize_params(optimizer)

    def _forward_pass_call(self, input, training=True):
        o = self.custom_layer1._forward_pass(input, training=training)
        o = self.custom_layer2._forward_pass(o, training=training)
        o = self.act._forward_pass(o)
        return o

optimizer = Adam()
model = ANNModel(optimizer)

epochs = 100

for i in range(epochs):
    for X_batch, y_batch in batch_iterator(X, y, batch_size=25):
        X_t = R.t(X_batch)
        y_t = R.t(y_batch)

        out = model._forward_pass(X_t)
        loss = R.square_loss(y_t, out)
        model._backward_pass(loss)

out = model._forward_pass(R.t(X_test), training=False)
out.persist_op(name="prediction")

R.activate()
R.execute(participants=1)
R.track_progress()

prediction = R.fetch_persisting_op(op_name="prediction")
y_pred = np.argmax(prediction['result'].detach().numpy(), axis=1)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)