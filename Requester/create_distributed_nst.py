from ctypes.wintypes import POINT
from dotenv import load_dotenv
load_dotenv()

import os
import ravop as R

from examples import nst
import numpy as np


content_layers = ['Conv2D_9']
style_layers = ['Conv2D_1',
                'Conv2D_3',
                'Conv2D_5',
                'Conv2D_8',
                'Conv2D_11']



# Initialize and create graph
R.initialize(ravenverse_token=os.environ.get("TOKEN"))
# R.initialize(ravenverse_token="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNjY1NzgzOTgwLCJpYXQiOjE2NjMwNjI5MzMsImp0aSI6ImVhMDBiM2Q3NTE0NjRlNDZhMGYzYzg3ZTUzMmRkZjNhIiwidXNlcl9pZCI6Ijc4MjA4MjgwNDUifQ.xAY2HFgzFYASKVBubDV0fgQOfWeNQuZai6eb_vJl89M")

R.flush()
R.Graph(name='nst', algorithm='neural_style_transfer', approach='distributed')

# dataset
X_train, X_test, y_train, y_test = nst.get_dataset()
# X_train, X_test, y_train, y_test = np.random.random((224,224,3)),np.random.random((224,224,3)),list([1]),list([1])


# create cnn model
model = nst.create_model()
po=[]
for _ in range(30):
    layer_vals=nst.forward_p(model,X_train)
    content_layers = ['Conv2D_9']
    style_layers = ['Conv2D_1',
                'Conv2D_3',
                'Conv2D_5',
                'Conv2D_8',
                'Conv2D_11']
    content_layer_out=[]
    style_layer_out=[]
    for con in content_layers:
        content_layer_out.append(layer_vals[con])
        layer_vals[con].persist_op(name =con+"_iter"+str(_))
    for sty in style_layers:
        style_layer_out.append(layer_vals[sty])
        layer_vals[sty].persist_op(name = sty+"_iter"+str(_))
    # conv2d_1=layer_vals['Conv2D_1']
    # conv2d_1.persist_op(name = "val_{}_batch_{}".format('conv1',_))
    # Activation_10_relu=layer_vals['Activation_10_relu']
    # Activation_10_relu.persist_op(name = "val_{}_batch_{}".format('activ_10_',_))
    print(layer_vals)



# compile it and start the execution
nst.compile()
nst.execute()

# cnn.get_score()
# print("Loss: ", loss)
# print("Accuracy: ", acc)



cont,st=[],[]
for _ in range(3):
    for con in content_layers:    
        cont.append( R.fetch_persisting_op(op_name=con+"_iter"+str(_)))
    for sty in style_layers:
        st .append(R.fetch_persisting_op(op_name=sty+"_iter"+str(_)))
# print(conv,act)
print( np.shape(cont[0]['result']), np.shape(st[0]['result']),np.shape(cont[1]['result']), np.shape(st[1]['result']))#,np.shape(conv[2]['result']), np.shape(act[2]['result']))
print("____________________________________________________________________")
print(cont[0].keys())