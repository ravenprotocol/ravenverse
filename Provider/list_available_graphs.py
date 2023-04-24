import os

from dotenv import load_dotenv

load_dotenv()

from ravpy.utils import list_graphs

if __name__ == '__main__':
    list_graphs(approach="distributed")
