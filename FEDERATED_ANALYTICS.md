# Federated Analytics

Federated Analytics is a feature of the Raven Distribution Framework that enables secure dynamic aggregation of statistics such as mean, variance and standard deviation across data that is hosted privately on multiple clients. 

## Usage

### 1. Configure RDF

Make sure RDF is configured correctly and Ravsock server is up and running : [Instructions](README.md)

    
### 2. Developer Side

Create a federated analytics graph by providing its name, approach and rules which clients must adhere to. 


    graph = R.Graph(name="Office Data", approach="federated",
                rules=json.dumps({"rules": {"age": {"min": 18, "max": 80},
                                            "salary": {"min": 1, "max": 5},
                                            "bonus": {"min": 0, "max": 10},
                                            "fund": {}
                                            },
                                  "max_clients": 1})) 

- Name: The name for the graph set by the developer. Preferably a meaningful name which allows clients to identify the type of dataset desired by the developer.

- Approach: Set to 'federated'.

- Rules: The rules dictionary must contain the names of all the columns of data required by the developer for aggregation and their corresponding constraints as shown above. The clients will then be able to filter their data accordingly. 
Note: An empty dictionary for a column signifies no constraints. All values in that column shall be considered.

- Max Clients: The number of clients whose data must be aggregated and returned to the developer.

### Creation of Federated Ops 
The following code snippet creates and adds ops to the previously declared graph. 
    
    mean = R.federated_mean() 
    variance = R.federated_variance() 
    standard_deviation = R.federated_standard_deviation()
    
The results of aggregation can be fetched by calling the afforementioned ops.

    # Wait for the results
    print("\nAggregated Mean: ", mean())
    print("\nAggregated Variance: ", variance())
    print("\nAggregated Standard Deviation: ", standard_deviation())

The results will be ready once ```max_clients``` number of clients have participated.

Note: The proper way of wrapping up ops in a graph is by calling ```graph.end()``` at the end of the code. This checks for any failed ops and lets the developer know.

#### Sample Test Code

    python federated_test.py

### 3. Client Side

As of now, Federated Analytics is natively supported by Raven's Python Clients (ravpy).

Upon configuration, RDF ensures that ravpy gets properly installed. 

For a client to view the available pending graphs and it's corresponding data rules: 

    python run_client.py --action list

The client must note the ```graph_id``` for the graph in which it wants to participate.

For the client to participate in it's desired graph: 

    python run_client.py --action participate --cid 123 --federated_id <graph_id>

Note: The ```cid``` argument is a unique username provided by the client. 

The terminal will then prompt the client to provide the path for it's dataset. 

The data can be placed inside ```/ravpy/data/``` folder. The data must be a ```.csv``` file containing atleast all columns mentioned in the graph's rules in any order. 
