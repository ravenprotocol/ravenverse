FROM ubuntu:18.04

RUN apt-get update

RUN apt-get -y install python3-pip

COPY . /

RUN pip3 install -r requirements.txt

RUN apt-get install sqlite3 libsqlite3-dev

EXPOSE 9999

RUN python3 setup_n_clean.py

ENTRYPOINT python3 run_socket_server.py

