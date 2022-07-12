from dotenv import load_dotenv

load_dotenv()

import pickle as pkl
import os
import ravop as R

R.initialize(ravenverse_token=os.environ.get("TOKEN"))

test_model = pkl.load(open("model.pkl", "rb"))
print("\n\n Pickle loaded model: \n")
test_model.summary()

test_model.get_onnx_model("test_ann")