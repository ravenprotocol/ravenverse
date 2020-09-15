from ravop.socket_client import SocketClient
import app.models as models
import numpy as np
import json


def start_operation(data1, data2, operation):
    if operation == 1:
        result = data1.matmul(data2)
    elif operation == 2:
        result = data1.add(data2)
    elif operation == 3:
        result = data1.sub(data2)
    elif operation == 4:
        result = data1.elemul(data2)
    elif operation == 5:
        result = data1.div(data2)
    elif operation == 6:
        result = data1.linear()
    elif operation == 7:
        result = data1.neg()
    elif operation == 8:
        result = data1.exp()
    elif operation == 9:
        result = data1.trans()
    elif operation == 10:
        result = data1.natlog()
    elif operation == 11:
        result = data1.matsum()
    else:
        result = None

    socket_client = SocketClient().connect()
    socket_client.emit("update_server", data=None, namespace="/ravop")
    return result
