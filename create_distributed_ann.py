from dotenv import load_dotenv
load_dotenv()

import os
import ravop as R
from examples import ann

# Initialize and create graph
R.initialize(ravenverse_token=os.environ.get("TOKEN"))
R.flush()
R.Graph(name='ann', algorithm='neural_network', approach='distributed')

# dataset
X, X_test, y, y_test, n_hidden, n_features = ann.get_dataset()

# create ann model
model = ann.create_model(n_hidden, n_features)

# start training
model = ann.train(model, X, y, n_epochs=1)

# test the model
ann.test(model, X_test)

# compile it and start the execution
ann.compile()
ann.execute()

# accuracy score
accuracy = ann.score(y_test)
print("Accuracy:", accuracy)
