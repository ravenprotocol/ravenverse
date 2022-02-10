import json

import ravop as R

# Create a graph to calculate mean, variance and standard deviation using federated learning approach
graph = R.Graph(name="Office Data", approach="federated",
                rules=json.dumps({"rules": {"age": {"min": 18, "max": 80},
                                            "salary": {"min": 1, "max": 5},
                                            "bonus": {"min": 0, "max": 10},
                                            "fund": {}
                                            },
                                  "max_clients": 1}))

mean = R.federated_mean()
variance = R.federated_variance()
standard_deviation = R.federated_standard_deviation()

# Wait for the results
print("\nAggregated Mean: ", mean())
print("\nAggregated Variance: ", variance())
print("\nAggregated Standard Deviation: ", standard_deviation())

# End graph and quit
graph.end()
