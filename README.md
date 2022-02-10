<div align="center">
  <img src="https://static.wixstatic.com/media/8e555b_b0053aa9f21e4ff2bed34105ef06189d~mv2_d_4703_2828_s_4_2.png/v1/fill/w_156,h_86,al_c,q_85,usm_0.66_1.00_0.01/RP-Logo-B.webp">
<h1> Raven Distribution Framework(RDF) </h1>
</div>


[Raven Distribution Framework](https://www.ravenprotocol.com)


### What is Raven Distribution Framework?
The foundation for any Machine Learning or Deep Learning Framework. Simply put, it is more like a decentralized calculator, comparable to a decentralized version of the IBM machines that were used to launch the Apollo astronauts. Apart from building ML/DL frameworks, a lot more can be done on it, such as maximizing yield on your favorite DeFi protocols like Compound and more!


### Setup 
    
    # Create a virutal environment before you install RDF libraries
    # Set up everything and install dependencies
    sh setup.sh
    
    
### Start ravsock

    python3 run_ravsock.py
   
### Create a federated analytics graph and create federated ops

    Kindly visit TEST_FEDERATED_ANALYTICS.md for this
    python3 federated_test.py
   
### Start a federated client

    # Pass client id and federated graph id to join
    python3 run_client.py --action participate --cid 111 --federated_id 1

### How to contribute:

Step 1: Fork

Step 2: Write your code

Step 3: Create a pull request


### License

<a href="https://github.com/ravenprotocol/raven-distribution-framework/blob/master/LICENSE"><img src="https://img.shields.io/github/license/ravenprotocol/raven-distribution-framework"></a>
