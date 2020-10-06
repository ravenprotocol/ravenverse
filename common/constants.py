import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILES_PATH = os.path.join(BASE_DIR, "files")

REDIS_HOST = "localhost"
REDIS_PORT = "6379"
REDIS_DB = "0"

MYSQL_HOST = "146.148.39.78"
MYSQL_PORT = "3306"
MYSQL_USER = "raven"
MYSQL_PASSWORD = ""
MYSQL_DATABASE = "ravenwebdemo"
