import contextlib
import subprocess
import time

import sqlalchemy

MYSQL_USER = "root"
MYSQL_PASSWORD = ""
MYSQL_DATABASE = "ravenwebdemo"
MYSQL_HOST = "0.0.0.0"
MYSQL_PORT = "3306"

container_name = "rwd_mysql"

with open("../output.log", "a") as output:
    subprocess.call("docker container stop {}".format(container_name), shell=True, stdout=output, stderr=output)
    subprocess.call("docker container rm {}".format(container_name), shell=True, stdout=output, stderr=output)
    subprocess.call("docker run --name={} --env='MYSQL_ROOT_PASSWORD={}' "
                    "-p {}:3306 -d mysql:latest".format(container_name, MYSQL_PASSWORD, MYSQL_PORT),
                    shell=True, stdout=output, stderr=output)

time.sleep(20)

with sqlalchemy.create_engine('mysql://{}:{}@{}:{}/{}'.format(MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST, MYSQL_PORT, "mysql"),
                              isolation_level='AUTOCOMMIT').connect() as connection:
    with contextlib.suppress(sqlalchemy.exc.ProgrammingError):
        connection.execute("CREATE DATABASE {}".format(MYSQL_DATABASE))
        print("Database created")
