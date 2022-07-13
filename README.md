<div align="center">
  <img src="https://user-images.githubusercontent.com/36446402/178711421-1b2de4f7-5f6a-48f9-af26-5d2d7a2306dd.png" width="100" height="55">
<h1> Ravenverse </h1>
</div>

## What is [Ravenverse](https://www.ravenprotocol.com)?
The foundation for any Machine Learning or Deep Learning Framework. Simply put, it is more like a decentralized calculator, comparable to a decentralized version of the IBM machines that were used to launch the Apollo astronauts. Apart from building ML/DL frameworks, a lot more can be done on it, such as maximizing yield on your favorite DeFi protocols like Compound and more!

<!-- ![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/solar.png)

## Features
 -->

The Ravenverse is a community-developed implementation of the decentralized computing model outlined by [Raven Protocol](https://www.ravenprotocol.com/). Within the Raven ecosystem today, there are two main actors: 

* **Developers:** Create models that need to be trained
* **Contributors:** Provide computational power to train the models


### Core libraries

* ```RavOp```: Core operations models for distributed computation
* ```RavSock```: Socket server to moderate client connections
* ```RavFTP```: FTP server to facilitate the transfer of files

### Libraries built on top of core libraries
* ```RavML```: Machine learning specific library
* ```RavDL```: Deep learning specific library
* ```RavViz``` (coming soon): A dashboard to visualize operations and client connections

### Client libraries
* ```Ravpy```: Python client for federated analytics and distributed computing
* ```RavJS```: Javascript library to retrieve and calculate operations

The current release of Ravenverse supports **Distributed Computing** and **Federated Analytics** across a network of Developers and Contributors around the world.

## What is Distributed Computing

To share and coordinate their processing capacity for a shared computational need, such as the training of a sizable Machine Learning model, distributed computing connects diverse computing resources, such as PCs and cellphones. Each of these resources or nodes receives some data and completes a portion of a task through communication with a server and, in some situations, with other nodes. These nodes can coordinate their operations to quickly and effectively meet a large-scale, complicated computational requirement.

## What is Federated Analytics

Key statistics like mean, variance, and standard deviation may be generated across numerous private datasets using federated analytics, a novel method of data analysis, without jeopardising privacy. It functions similarly to federated learning in that it does local computations on the data from each client device and only provides developers with the aggregated results—never any data from a specific device. Without leaving the premises, sensitive data can be examined, including financial transactions, employee information, and medical records.

Detailed walkthroughs (from both Developer's and Contributor's perspectives) on setting up and testing both distributed computing and federated analytics driven algorithms can be found under *```Tutorials```* folder.  

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## Setup 

This repository can be used by both developers and contributors as a foundation for setting up Raven's various other libraries and running their algorithms.
   
### Create a Virtual Environment

    conda create -n <env_name> python=3.8
    conda activate <env_name>

### Clone the Repository

    git clone https://github.com/ravenprotocol/ravenverse.git

### Getting your Ravenverse Token

Visit the Ravenverse Website (preferably on Chrome) and login using your MetaMask Wallet Credentials.

Copy your Private Ravenverse Token. This token will be required by all Raven libraries to authenticate transactions and data transfers.


### For Developers

Install our python packages hosted on PyPi.

    pip install ravop
    pip install ravdl
    pip install ravml

With these libraries, Developers can build, train and test their ML/DL models. 


### For Contributors

If your wish to enter Ravenverse as a Contributor, you must install RavPy:

    pip install ravpy

RavPy is a python SDK that allows contributors to intuitively participate in any ongoing graph computations in the Ravenverse.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)


## How to contribute:

Contributions are what make the open source community such a wonderful place to learn, be inspired, and create. You may contribute to our individual [Repositories](https://github.com/ravenprotocol). Please read our [Contributor Guide](CONTRIBUTING.md). 

Any help you can give is much appreciated.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

## License

<a href="https://github.com/ravenprotocol/ravenverse/blob/master/LICENSE"><img src="https://img.shields.io/github/license/ravenprotocol/ravenverse"></a>

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details