## How to test federated analytics
How to use federated analytics to calculate mean, variance and standard deviation

### Steps

#### 1. Clone ravsock, ravop, ravjs, ravftp and ravpy

    
    git clone https://github.com/ravenprotocol/ravsock.git
    git clone https://github.com/ravenprotocol/ravjs.git
    git clone https://github.com/ravenprotocol/ravop.git
    git clone https://github.com/ravenprotocol/ravpy.git
    git clone https://github.com/ravenprotocol/ravftp.git

#### 2. Setup dependencies(ravjs, ravop, ravftp and ravpy)


    sh setup.sh
    
#### 3. Set FTP server directory and FTP environ path inside of ravsock/config.py


    # ~/ravftp
    FTP_SERVER_DIR = "<ravftp_dir>"
    
    # ~/miniconda3/envs/ravftp/bin
    FTP_ENVIRON_DIR = "<ravftp_virtual_env_dir>"
    
#### 4. Set RDF_DATABASE_URI and create database with database tables required for the project


    RDF_DATABASE_URI = "sqlite:///rdf.db?check_same_thread=False"

    python reset.py   # Deletes the old database 
    
    
#### 5. Start ravsock server (socket + http server)


    python run.py
    
    
#### 6. Create a federated analytics graph by providing its name, approach and rules which clients must adhere to 


    graph = R.Graph(name="Office Data", approach="federated",
                rules=json.dumps({"rules": {"age": {"min": 18, "max": 80},
                                            "salary": {"min": 1, "max": 5},
                                            "bonus": {"min": 0, "max": 10},
                                            "fund": {}
                                            },
                                  "max_clients": 1})) 
    
#### 7. Now you can calculate mean, variance or standard deviation by following the steps given below

    
    mean = R.federated_mean() 
    
    or 
    
    variance = R.federated_variance()
    
    or 
    
    standard_deviation = R.federated_standard_deviation()
    
