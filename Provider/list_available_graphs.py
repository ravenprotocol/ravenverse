import os

from dotenv import load_dotenv

load_dotenv()

from ravpy.initialize import initialize
from ravpy.utils import list_graphs

if __name__ == '__main__':
    client = initialize(os.environ.get("TOKEN"))
    list_graphs(approach="distributed")
