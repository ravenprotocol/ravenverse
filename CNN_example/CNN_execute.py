from dotenv import load_dotenv

load_dotenv()
import os
import ravop as R

R.initialize(ravenverse_token=os.environ.get("TOKEN"))

R.execute()
R.track_progress()

output = R.fetch_persisting_op(op_name="training_loss_epoch_1_batch_1")
print("training_loss_epoch_1_batch_1: ", output)
