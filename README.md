## Raven Distribution Framework
https://www.ravenprotocol.com/

#### Installation Instructions

1. Install dependencies - pip3 install -r requirements.txt
2. Setup and clean - python3 setup_n_clean.py
3. Run socket server - python3 run_socket_server.py
4. Open index.html in your browser
5. Run our application - python3 run_app.py


#### Docker Implementation

Create a docker image
        
    sudo docker build -t raven .
    
Stop and remove all containers
    
    sudo docker stop $(sudo docker ps -aq)
    sudo docker rm $(sudo docker ps -aq)

Create and start the docker container

    sudo docker run --name raven_server -p 9999:9999 raven
    
Stop and remove the container

    sudo docker container stop raven_server
    sudo docker container rm raven_server
    
Remove an image

    sudo docker rmi raven
    
Enter into the container

    sudo docker exec -it raven_server bash

#### How to contribute:

Step 1: Fork

Step 2: Write your code

Step 3: Create a pull request

#### License
[MIT License](https://github.com/ravenprotocol/raven-distribution-framework/blob/master/LICENSE)
