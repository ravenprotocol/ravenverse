<div align="center">
  <img src="https://static.wixstatic.com/media/8e555b_b0053aa9f21e4ff2bed34105ef06189d~mv2_d_4703_2828_s_4_2.png/v1/fill/w_156,h_86,al_c,q_85,usm_0.66_1.00_0.01/RP-Logo-B.webp">
</div>

## Raven Distribution Framework(RDF) 
https://www.ravenprotocol.com/


#### What is Raven Distribution Framework?
The foundation for any Machine Learning or Deep Learning Framework. Simply put, it is more like a decentralized calculator, comparable to a decentralized version of the IBM machines that were used to launch the Apollo astronauts. Apart from building ML/DL frameworks, a lot more can be done on it, such as maximizing yield on your favorite DeFi protocols like Compound and more!


#### Setup

1. Install and create a virtual environment
     
     ```
     pip3 install virtualenv & virtualenv venv -p python3 & source venv/bin/activate
     ```

2. Install dependencies

     ```
     pip3 install -r requirements.txt
     ```

3. Run socket server

     ```
     python3 run_socket_server.py
     ```

4. Install redis-server

    ```
    # Ubuntu
    sudo apt install redis-server

    # Mac
    brew install redis
    brew services start redis
    ```

5. Specify MySQL database credentials in the common/constants.py file

     ```
     MYSQL_HOST = "localhost"
     MYSQL_PORT = "3306"
     MYSQL_USER = "root"
     MYSQL_PASSWORD = "password"
     MYSQL_DATABASE = "rdf"
     ```

4. Open ravenclient/index.html in your browser

    ```
    ravenclient/index.html
    ```

#### How to write operations?

     from ravop.core import Scalar, Tensor
     a = Scalar(10)
     b = Scalar(20)

     # Calculate sum of a and b
     c = a.add(b)

     # Subtract b from a
     d = a.sub(b)

     # Matrix Multiplication
     a = Tensor([[2,3,4],[3,4,5],[5,6,7]])
     b = Tensor([[3],[4],[6]])
     c = a.matmul(b)

#### How to contribute:

Step 1: Fork

Step 2: Write your code

Step 3: Create a pull request

#### License
[MIT License](https://github.com/ravenprotocol/raven-distribution-framework/blob/master/LICENSE)
