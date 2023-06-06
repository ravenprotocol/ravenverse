'''
The following implementation is based on the Sort problem solution by Andrej Karpathy:
https://github.com/karpathy/minGPT/blob/master/demo.ipynb
'''

from dotenv import load_dotenv
load_dotenv()

import os
import ravop as R
import numpy as np
from ravdl.v2 import Pytorch_Model
import random
import pickle
from ravdl.v2.optimizers import Adam
from ravdl.v2.utils.data_manipulation import batch_iterator

# Initialize and create graph
R.initialize(ravenverse_token=os.environ.get("TOKEN"))
R.flush()
R.Graph(name='sorter_gpt_model', algorithm='pytorch_model', approach='distributed')

split = 'train'

X_train = []
y_train = []

for i in range(12800):

    while True:
        inp = np.random.randint(0, 3, size=(6,))

        if random.uniform(0,1) < 0.5:
            if np.unique(inp).size > 6 // 2:
                continue

        h = hash(pickle.dumps(inp.tolist()))
        inp_split = 'test' if h % 4 == 0 else 'train' # designate 25% of examples as test
        if inp_split == split:
            break

    sol = np.sort(inp)
    cat = np.concatenate((inp, sol), axis=0)

    x = cat[:-1].copy()
    y = cat[1:].copy()
    y[:6-1] = -1

    X_train.append(x)   
    y_train.append(y)

X_train = np.array(X_train)
y_train = np.array(y_train)

n_epochs = 1
optimizer = Adam()

model_op = R.model('generate_torchscript_models/sorter_gpt.pt')

model = Pytorch_Model(model_op=model_op)
model.initialize(optimizer)


for i in range(n_epochs):
    for X_batch, y_batch in batch_iterator(X_train, y_train, batch_size=64):
        op = model._forward_pass(R.t(X_batch), training=True)
        loss = R.cross_entropy_loss(R.t(y_batch), op, ignore_index=-1, reshape_target=(-1,3), reshape_label=(-1,))
        model._backward_pass(loss)

test_input = np.array([0, 2, 1, 2, 1, 0])

def generate(x, steps=6):
    x = np.expand_dims(x,axis=0)
    x = R.t(x)

    for i in range(steps):
        outputs = model._forward_pass(x, training=False)
        output_tensor = R.index(outputs, indices=str({"indices":"[:,-1,:]"}))
        output_token = R.argmax(output_tensor, axis=-1, keepdims="True")
        x = R.concat(x,output_token,axis=-1)

    x.persist_op('outputs')


output = generate(test_input, 6)


R.activate()
R.execute()
R.track_progress()

output = R.fetch_persisting_op('outputs')
output = np.array(output)[:,6:]

print('Test Input: ', test_input)
print('Predicted: ', output)