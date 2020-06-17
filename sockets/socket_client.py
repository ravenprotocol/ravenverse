import socketio

from settings import RAVSOCK_SERVER_URL


class RavOPNamespace(socketio.ClientNamespace):
    def on_connect(self):
        print('Connected to the server')

    def on_disconnect(self):
        print('Disconnected from the server')

    def on_message(self, data):
        print('Message received:', data)

    def on_result(self, data):
        print(data)


class SocketClient(object):
    def __init__(self):
        self.client = None

    def connect(self):
        self.client = socketio.Client()
        self.client.register_namespace(RavOPNamespace('/ravop'))
        self.client.connect(RAVSOCK_SERVER_URL+"?client_name=ravop")

        return self.client
