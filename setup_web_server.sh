#!/bin/bash
sudo apt-get install python3-pip
sudo pip3 install virtualenv
virtualenv venv -p python3
venv/bin/pip install -r requirements.txt
venv/bin/python manage.py runserver 0.0.0.0:8000
