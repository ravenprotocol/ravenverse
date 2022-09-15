import os
from dotenv import load_dotenv
load_dotenv()

import json
import ravop as R

R.initialize(ravenverse_token=os.environ.get("TOKEN"))

R.flush()
R.Graph(name="Office Data", approach="federated",
                rules=json.dumps({"rules": {"age": {"min": 18, "max": 80},
                                            "salary": {"min": 1, "max": 5},
                                            "bonus": {"min": 0, "max": 10},
                                            "fund": {}
                                            },
                                  "max_clients": 1}))

mean = R.federated_mean()
variance = R.federated_variance()
standard_deviation = R.federated_standard_deviation()

mean.persist_op(name="mean")
variance.persist_op(name="variance")
standard_deviation.persist_op(name="standard_deviation")

R.activate()

R.execute()
R.track_progress()

mean_output = R.fetch_persisting_op(op_name="mean")
print("\n\nMean: ", mean_output)

variance_output = R.fetch_persisting_op(op_name="variance")
print("\n\nVariance: ", variance_output)

standard_deviation_output = R.fetch_persisting_op(op_name="standard_deviation")
print("\n\nStandard_deviation: ", standard_deviation_output)