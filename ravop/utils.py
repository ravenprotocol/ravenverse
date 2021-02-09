from ravop.socket_client import SocketClient


def inform_server():
    socket_client = SocketClient().connect()
    socket_client.emit("inform_server", data={"type": "event"}, namespace="/ravop")
