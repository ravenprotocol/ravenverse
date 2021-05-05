from aiohttp import web

# We kick off our server
from ravsock.socketio_server import app

if __name__ == '__main__':
    print("Starting server...")
    web.run_app(app, port=9999)
