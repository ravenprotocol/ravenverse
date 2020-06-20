FROM ubuntu:18.04

COPY . /

EXPOSE 9999

RUN bash install_deps.sh

ENTRYPOINT python3 run_socket_server.py
