from dotenv import load_dotenv
load_dotenv()

import os
import ravop as R

from .examples import cnn

# Initialize and create graph
R.initialize(ravenverse_token=os.environ.get("TOKEN"))
R.flush()
R.Graph(name='cnn', algorithm='convolutional_neural_network', approach='distributed')

# dataset
X_train, X_test, y_train, y_test = cnn.get_dataset()

# create cnn model
model = cnn.create_model(X_test=X_test, y_test=y_test)

# start training
model = cnn.train(model, X_train, y_train, n_epochs=10)

cnn.test(model, X_test, y_test)

# compile it and start the execution
cnn.compile()
cnn.execute()

loss, acc = cnn.get_score()
print("Loss: ", loss)
print("Accuracy: ", acc)