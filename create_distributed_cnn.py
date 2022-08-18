from dotenv import load_dotenv
load_dotenv()

import os
import ravop as R
from examples import cnn

# Initialize and create graph
R.initialize(ravenverse_token=os.environ.get("TOKEN"))
R.flush()
R.Graph(name='cnn', algorithm='convolutional_neural_network', approach='distributed')

# dataset
X_train, X_test, y_train, y_test = cnn.get_dataset()

# create cnn model
model = cnn.create_model(X_test=X_test, y_test=y_test)

# start training
cnn.train(model, X_train, y_train)

# compile it and start the execution
cnn.compile()
cnn.execute()
