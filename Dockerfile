FROM python:3.8
ENV PYTHONUNBUFFERED 1
RUN mkdir /rdf
WORKDIR /rdf
ADD requirements.txt /rdf/
RUN pip install --upgrade pip && pip install -r requirements.txt
ADD . /rdf/
