import os

from dotenv import load_dotenv

load_dotenv()

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import ravop as R

R.initialize(ravenverse_token=os.environ.get("TOKEN"))

R.execute()
R.track_progress()

output = R.fetch_persisting_op(op_name="training_loss_epoch_1_batch_1")
print("training_loss_epoch_1_batch_1: ", output)


data = datasets.load_iris()
X = data.data
y = data.target
X= normalize(X,axis=0)
_, _, _ , y_test = train_test_split(X, y, test_size=0.33, random_state=42)

prediction = R.fetch_persisting_op(op_name="test_prediction")
y_pred = np.argmax(prediction,axis=1)

accuracy = accuracy_score(y_test, y_pred)

print ("Accuracy:", accuracy)
