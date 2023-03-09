from dotenv import load_dotenv
load_dotenv()

import numpy as np
import os
import ravop as R

from ravdl.v2.layers import Activation, Dense, BatchNormalization1D,BatchNormalization2D, Dropout, Conv2D, Flatten, MaxPooling2D
from ravdl.v2 import Functional
from ravdl.v2.optimizers import Adam, RMSprop
from ravdl.v2.utils.data_manipulation import batch_iterator
from utils import cnn

# Initialize and create graph
R.initialize(ravenverse_token=os.environ.get("TOKEN"))
R.flush()
R.Graph(name='cnn', algorithm='convolutional_neural_network', approach='distributed')

X, X_test, y, y_test = cnn.get_dataset()


class CNNModel(Functional):
    def __init__(self, optimizer):
        super().__init__()
        self.conv2d_1 = Conv2D(in_channels=1, out_channels=16, kernel_size=(3, 3), padding='same')
        self.act_1 = Activation('relu')
        self.maxpool2d_1 = MaxPooling2D(kernel_size=(2, 2), stride=2)
        self.drp_1 = Dropout(0.25)
        self.bn_1 = BatchNormalization2D(16)
        self.maxpool2d_2 = MaxPooling2D(kernel_size=(2, 2), stride=2)
        self.conv2d_2 = Conv2D(in_channels=16, out_channels=32, kernel_size=(3, 3), padding='same')
        self.act_2 = Activation('relu')
        self.maxpool2d_3 = MaxPooling2D(kernel_size=(2, 2), stride=2)
        self.drp_2 = Dropout(0.25)
        self.bn_2 = BatchNormalization2D(32)
        self.flatten = Flatten()
        self.dense_1 = Dense(256)
        self.act_3 = Activation('relu')
        self.drp_3 = Dropout(0.4)
        self.bn_3 = BatchNormalization1D()
        self.dense_2 = Dense(10)
        self.act_4 = Activation('softmax')

        self.initialize_params(optimizer)

    def _forward_pass_call(self, input, training=True):
        out = self.conv2d_1._forward_pass(input)
        out = self.act_1._forward_pass(out)
        out = self.maxpool2d_1._forward_pass(out)
        out = self.drp_1._forward_pass(out, training=training)
        out = self.bn_1._forward_pass(out, training=training)
        out = self.maxpool2d_2._forward_pass(out)
        out = self.conv2d_2._forward_pass(out)
        out = self.act_2._forward_pass(out)
        out = self.maxpool2d_3._forward_pass(out)
        out = self.drp_2._forward_pass(out, training=training)
        out = self.bn_2._forward_pass(out, training=training)
        out = self.flatten._forward_pass(out)
        out = self.dense_1._forward_pass(out)
        out = self.act_3._forward_pass(out)
        out = self.drp_3._forward_pass(out, training=training)
        out = self.bn_3._forward_pass(out, training=training)
        out = self.dense_2._forward_pass(out)
        out = self.act_4._forward_pass(out)

        return out
        

optimizer = Adam()
model = CNNModel(optimizer)

epochs = 50

for i in range(epochs):
    for X_batch, y_batch in batch_iterator(X, y, batch_size=256):
        X_t = R.t(X_batch)
        y_t = R.t(y_batch)

        out = model._forward_pass(X_t)
        loss = R.cross_entropy_loss(y_t, out)
        model._backward_pass(loss)

out = model._forward_pass(R.t(X_test), training=False)
out.persist_op(name="prediction")

R.activate()
R.execute(participants=1)
R.track_progress()

prediction = R.fetch_persisting_op(op_name="prediction")
y_pred = np.argmax(prediction['result'].detach().numpy(), axis=-1)
y_test = np.argmax(y_test, axis=-1)

accuracy = np.sum(y_pred == y_test, axis=0) / len(y_test)

print("Accuracy:", accuracy)