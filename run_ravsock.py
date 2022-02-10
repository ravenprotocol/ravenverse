from aiohttp import web

from ravsock.globals import globals as g
from ravsock.utils import create_database, create_tables
from ravsock.bindings import *

sio = g.sio
app = g.app

if __name__ == "__main__":
    print("Starting server...")

    print(sio)

    # Create database if not exists
    if create_database():
        create_tables()

    web.run_app(app, port=9999)
