from dotenv import load_dotenv
load_dotenv()
import os

from ravdl.v2.optimizers import Adam
from ravdl.v2.load_onnx_model import load_onnx
import ravop as R
from examples import ann

R.initialize(ravenverse_token=os.environ.get("TOKEN"))
R.flush()
R.Graph(name='ann', algorithm='neural_network', approach='distributed')

model = load_onnx(model_file_path="ann.onnx", optimizer=Adam(), loss='CrossEntropy')
model.summary()

X, X_test, y, y_test, n_hidden, n_features = ann.get_dataset()

# test the model
ann.test(model, X_test)

# compile it and start the execution
ann.compile()
ann.execute()

# accuracy score
accuracy = ann.score(y_test)
print("Accuracy:", accuracy)
