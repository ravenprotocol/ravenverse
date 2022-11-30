from ctypes.wintypes import POINT
from dotenv import load_dotenv
load_dotenv()

import os
import ravop as R

from examples import imgnet
import numpy as np

# Initialize and create graph
R.initialize(ravenverse_token=os.environ.get("TOKEN"))

R.flush()
R.Graph(name='imagenet', algorithm='imgnet', approach='distributed')

'''
Imagenet training ----> 
'''

imagenette_map = { 
    "n01440764" : "tench",
    "n02102040" : "springer",
    "n02979186" : "casette_player",
    "n03000684" : "chain_saw",
    "n03028079" : "church",
    "n03394916" : "French_horn",
    "n03417042" : "garbage_truck",
    "n03425413" : "gas_pump",
    "n03445777" : "golf_ball",
    "n03888257" : "parachute"
}
path_train="examples/datasets/imagenette2-320/train/"
X_train, y_train = imgnet.get_dataset(path_train)

model = imgnet.create_model()
# start training
model = imgnet.train(model, np.array(X_train[:10]),y_train[:10],batch_size=10, n_epochs=1)

imgnet.compile()
imgnet.execute()
R.track_progress()
