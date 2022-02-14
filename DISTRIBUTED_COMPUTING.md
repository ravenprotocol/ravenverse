# Distributed Computing

Distributed Computing is a feature of Raven Distribution Framework that allows a developer to train machine learning/deep learning models in a decentralized and distributed manner. It facilitates the faster and cheaper training of ML/DL models by splitting them into smaller groups of mathematical Ops and sending them to browser nodes (javascript clients) for local computation.

## Usage

### 1. Configure RDF

Make sure RDF is configured correctly and Ravsock server is up and running : [Instructions](README.md)

### 2. Developer Side

The Developer has to build their Model/Algorithm using the RavOp library that gets installed during RDF configuration. RavOp is our library to work with ops. You can create ops, interact with ops, and create scalars and tensors. 

[RavOp Documentation](https://github.com/ravenprotocol/docs/blob/master/docs/ravop.md)

    import ravop as R

    graph = R.Graph(name='test', approach='distributed')

    a = R.t([1, 2, 3])
    b = R.t([5, 22, 7])
    c = a + b

    print('c: ', c())

    graph.end()

The output of ```c()``` will get returned once the participating clients have calculated their assigned Ops.

The proper way of wrapping up ops in a graph is by calling ```graph.end()``` at the end of the code. This checks for any failed ops and lets the developer know.

A slightly more complex implementation can be found in ```distributed_test.py```

### 3. Client Side

As of this release, distributed computing is supported only by [RavJs](https://github.com/ravenprotocol/ravjs) Clients. The ravjs repository gets automatically cloned during RDF configuration.

- Make sure Ravsock server is up and running.

- In ```ravjs/raven.js``` file update the ```CID``` variable to a unique string before opening a new client. 

- On a new browser tab, open the following URL:

    http://localhost:9999/

- Once connected, click on ```Participate``` button. This triggers the execution of a local client benchmarking code and returns it's results to the server. The server utilizes this data for optimizing the scheduling algorithm.

The client will now dynamically receive groups of Ops from the server, compute them and return the results back to the server.  