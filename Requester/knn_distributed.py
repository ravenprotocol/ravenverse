from dotenv import load_dotenv
load_dotenv()

import os
import ravop as R
from utils import knn

# Initialize and create graph
R.initialize(ravenverse_token=os.environ.get("TOKEN"))
R.flush()
R.Graph(name='knn', algorithm='k-nearest neighbours', approach='distributed')


# dataset
X_train, X_test, y_train, y_test = knn.get_dataset()

# create knn model
model = knn.create_model()

# start training
model = knn.train(model, X_train, y_train)

# test the model
knn.test(model, X_test, y_test)

# compile it and start the execution
knn.compile()
knn.execute(participants=1)

# test_prediction and score
test_prediction = R.fetch_persisting_op(op_name="test_prediction")
score = R.fetch_persisting_op(op_name="score")
print("Test Prediction : ", test_prediction)
print("Score : ", score)