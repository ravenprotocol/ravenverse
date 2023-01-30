FROM python:3.7.11

ARG DEBIAN_FRONTEND=noninteractive

COPY . /

RUN python -m ensurepip --upgrade
RUN python -m pip install --upgrade pip

RUN pip install -r requirements.txt

CMD ["run_distributed_client.py"]
ENTRYPOINT ["python"]