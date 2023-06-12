from dotenv import load_dotenv
load_dotenv()

import os
import ravop as R
import numpy as np
from utils import cnn
from ravdl.v2 import Pytorch_Model
from ravdl.v2.optimizers import Adam
from ravdl.v2.utils.data_manipulation import batch_iterator
import torch

# Initialize and create graph
R.initialize(ravenverse_token=os.environ.get("TOKEN"))
R.flush()
R.Graph(name='test_pytorch_model', algorithm='pytorch_model', approach='distributed')#, gpu_required = "yes")

X, X_test, y, y_test = cnn.get_dataset()

model_op = R.model('test_model.pt')
optimizer = Adam()

model = Pytorch_Model(model_op=model_op)
model.initialize(optimizer)

epochs = 10

for i in range(epochs):
    for X_batch, y_batch in batch_iterator(X, y, batch_size=256):
        X_t = R.t(X_batch.astype(np.float32))
        y_t = R.t(y_batch.astype(np.float32))

        out = model._forward_pass(X_t)
        loss = R.square_loss(y_t, out)

        # Set step = True whenever optimizer step needs to be called after backprop (defaults to True).
        model._backward_pass(loss, step = True)

model.save_model(name='my_net')

test_input = R.t(X_test.astype(np.float32))

output = model._forward_pass(test_input, training=False)
output.persist_op("output")
R.activate()
R.execute()
R.track_progress()

prediction = R.fetch_persisting_op(op_name="output")
y_pred = np.argmax(prediction.detach().numpy(), axis=-1)
y_test = np.argmax(y_test, axis=-1)

accuracy = np.sum(y_pred == y_test, axis=0) / len(y_test)

print("Accuracy:", accuracy)

my_net = R.fetch_persisting_op(op_name="my_net")
my_net.eval()
out = my_net(torch.tensor(X_test.astype(np.float32)))
y_pred = np.argmax(out.detach().numpy(), axis=-1)
print(y_pred)
accuracy = np.sum(y_pred == y_test, axis=0) / len(y_test)

print("Accuracy of loaded model:", accuracy)
