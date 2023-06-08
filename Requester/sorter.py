from dotenv import load_dotenv
load_dotenv()

import os
import numpy as np
import random
import pickle

import ravop as R
from transformer import GPT
from ravdl.v2.optimizers import Adam
from ravdl.v2.utils.data_manipulation import batch_iterator


R.initialize(ravenverse_token=os.environ.get("TOKEN"))
R.flush()
R.Graph(name='gpt_train', algorithm='transformer', approach='distributed')

split = 'train'

X_train = []
y_train = []

for i in range(6400):

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

model = GPT(vocab_size=3, embed_dim=48, 
            block_size=11, n_heads=3, 
            n_layer=3, seq_length=11, 
            optimizer=optimizer)

sequence_length = 11

causal_mask = np.triu(np.ones((sequence_length,sequence_length)),k=1)

for i in range(n_epochs):
    for X_batch, y_batch in batch_iterator(X_train, y_train, batch_size=64):
        op = model._forward_pass(R.t(X_batch), training=True, batch_size=X_batch.shape[0], mask=causal_mask)
        loss = R.cross_entropy_loss(R.t(y_batch), op, ignore_index=-1, reshape_target=(-1,3), reshape_label=(-1,))
        model._backward_pass(loss)

test_input = np.array([1, 0, 2, 2, 1, 1])

def generate(x, steps=6):
    x = np.expand_dims(x,axis=0)
    x = R.t(x)
    
    causal_mask = np.triu(np.ones((6,6)),k=1)
    model.seq_length = 6
    for i in range(steps):
        outputs = model._forward_pass(x, training=False, batch_size=1, mask=causal_mask)
        output_tensor = R.index(outputs, indices=str({"indices":"[:,-1,:]"}))
        output_token = R.argmax(output_tensor, axis=-1, keepdims="True")
        x = R.concat(x,output_token,axis=-1)
        model.seq_length = model.seq_length + 1
        causal_mask = np.triu(np.ones((model.seq_length,model.seq_length)),k=1)

    x.persist_op('outputs')


output = generate(test_input, 6)


R.activate()
R.execute()
R.track_progress()

output = R.fetch_persisting_op('outputs')
output = np.array(output)[:,6:]

print('Test Input: ', test_input)
print('Predicted: ', output)