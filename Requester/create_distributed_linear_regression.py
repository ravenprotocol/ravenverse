from dotenv import load_dotenv
load_dotenv()

import os
import ravop as R
from .examples import linear_regression

# Initialize and create graph
R.initialize(ravenverse_token=os.environ.get("TOKEN"))
R.flush()
R.Graph(name='lin_reg', algorithm='linear_regression', approach='distributed')


# dataset
x, y, theta = linear_regression.get_dataset()

# create linear regression model
model = linear_regression.create_model(x, y, theta)

# start training
model = linear_regression.train(model)

# test the model
linear_regression.test(model, x, y)

# compile it and start the execution
linear_regression.compile()
linear_regression.execute()

# test_prediction and optimized theta
optimal_theta = R.fetch_persisting_op(op_name="theta")
pred = R.fetch_persisting_op(op_name="predicted values")
print("Optimized Theta", optimal_theta)
print("Predicted Values:", pred)
model.plot_graph(optimal_theta=optimal_theta)