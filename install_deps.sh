apt-get update
apt-get -y install python3-pip
pip3 install -r requirements.txt
apt-get install sqlite3 libsqlite3-dev
python3 setup_n_clean.py