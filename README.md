<div align="center">
  <img src="https://static.wixstatic.com/media/8e555b_b0053aa9f21e4ff2bed34105ef06189d~mv2_d_4703_2828_s_4_2.png/v1/fill/w_156,h_86,al_c,q_85,usm_0.66_1.00_0.01/RP-Logo-B.webp">
<h1> Raven Distribution Framework(RDF) </h1>
</div>

## What is [Raven Distribution Framework](https://www.ravenprotocol.com)?
The foundation for any Machine Learning or Deep Learning Framework. Simply put, it is more like a decentralized calculator, comparable to a decentralized version of the IBM machines that were used to launch the Apollo astronauts. Apart from building ML/DL frameworks, a lot more can be done on it, such as maximizing yield on your favorite DeFi protocols like Compound and more!

<!-- ![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/solar.png)

## Features
 -->

The Raven Distribution Framework (RDF) is a community-developed implementation of the decentralized computing model outlined by [Raven Protocol](https://www.ravenprotocol.com/). Within the Raven ecosystem today, there are two main actors: 

* **[Developers](Developers):** Create models that need to be trained
* **[Clients](Clients):** Provide computational power to train the models

For [Developers](Developers), there are three core libraries that drive the main function for computation distribution ([RavOP](RavOp), [RavSock](RavSock), and [RavFTP](RavFTP)), along with a growing list of libraries that extend the core to be more developer-friendly to use.

For [Clients](Clients), there are two libraries, [Ravpy](ravpy) is the python client for federated and distributed computing and the javascript library [RavJS](RavJS) enables anyone with a browser to contribute processing power. Additional clients are in consideration, such as Go and Rust (looking for community devs!).

### Core libraries

* [RavOP](RavOp): Core operations models for distributed computation
* [RavSock](RavSock): Socket server to moderate client connections
* [RavFTP](RavFTP): FTP server to facilitate transfer of files

### Libraries built on top of core libraries
* [RavML](RavML): Machine learning specific library
* RavDL (Coming soon): Deep learning specific library
* [RavViz](RavViz): A dashboard to visualize operations and client connections

### Client libraries
* [Ravpy](Ravpy): Python client for federated and distributed computing
* [RavJS](RavJS): Javascript library to retrieve and calculate operations

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/solar.png)

## Setup 
    
### Installation 
   
#### Create a virutal environment with Python 3.8 before you install RDF libraries
    conda create -n <env_name> python=3.8

#### Clone the Repository
    git clone https://github.com/ravenprotocol/raven-distribution-framework.git
#### Set up everything and install dependencies
    sh setup.sh

### Configure Paths
Navigate to ```ravsock/config.py``` and set the ```FTP_ENVIRON_DIR``` variable to the ```bin``` folder of your python virtual environment. For instance: 
    
    FTP_ENVIRON_DIR = "~/miniconda/envs/<env_name>/bin"

Note: Set ```ENCRYPTION = True``` in the same file if a layer of homomorphic encryption needs to be added for Federated Analytics.

Set ```RDF_DATABASE_URI``` in the same file.

    RDF_DATABASE_URI = "sqlite:///rdf.db?check_same_thread=False"

Create database with tables required for the project.

    python reset.py  

The server is now configured correctly and ready to be fired up.

### Start Ravsock Server

Ravsock is a crucial component of RDF that facilitates both federated and distributed functionalities of the framework. 

It sits between the developer(who creates ops and writes algorithms) and the contributor who contributes the idle computing power. It's scheduling algorithm oversees the distribution and statuses of different Ops, Graphs and Subgraphs across multiple Clients. 

    python3 run_ravsock.py


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/solar.png)


## How to Run

### Federated Analytics 

Kindly visit [FEDERATED_ANALYTICS.md](FEDERATED_ANALYTICS.md) for more info on creating and working with custom Federated Ops.

### Distributed Computing

Kindly visit [DISTRIBUTED_COMPUTING.md](DISTRIBUTED_COMPUTING.md) for more on creating graphs, initializing distributed clients in web browser and working with custom Ops to develop distributed ML algorithms.
   
![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/solar.png)


## How to contribute:

Contributions are what make the open source community such a wonderful place to learn, be inspired, and create. You may contribute to our individual [Repositories](https://github.com/ravenprotocol). 

- Fork

- Write your code

- Create a pull request

Any help you can give is much appreciated.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/solar.png)

## License

<a href="https://github.com/ravenprotocol/raven-distribution-framework/blob/master/LICENSE"><img src="https://img.shields.io/github/license/ravenprotocol/raven-distribution-framework"></a>

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details