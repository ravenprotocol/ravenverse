import datetime
import json
import logging
import logging.handlers
import pickle

import numpy as np
import socketio
from aiohttp import web

# creates a new Async Socket IO Server
from common.db_manager import DBManager, Client, Op, Data, OpStatus
from common.data_manager import DataManager
from .constants import RAVSOCK_LOG_FILE

# Set up a specific logger with our desired output level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add the log message handler to the logger
handler = logging.handlers.RotatingFileHandler(RAVSOCK_LOG_FILE)

logger.addHandler(handler)

sio = socketio.AsyncServer(cors_allowed_origins="*")

# Creates a new Aiohttp Web Application
app = web.Application()

# Binds our Socket.IO server to our Web App instance
sio.attach(app)

db = DBManager()
data_manager = DataManager(db)


# we can define aiohttp endpoints just as we normally
# would with no change
async def index(request):
    with open('../ravclient/index.html') as f:
        return web.Response(text=f.read(), content_type='text/html')


@sio.on('update_server', namespace="/ravop")
async def receive_op(sid):
    print("\nOp Received...", sid)

    """
    Steps:
    1. Get a pending op
    2. Get an idle client
    3. If both are none then
    """

    client_found = search_client()
    op_found = search_pending_op()

    if op_found is None and client_found is None:
        return

    print("Op:", op_found.id, )
    print("Client:", client_found.id, client_found.client_id)

    # Create payload
    inputs = json.loads(op_found.inputs)

    payload = create_payload(op_found.id, inputs, op_found.op_type, op_found.operator)

    await sio.emit("op", payload, namespace="/ravjs", room=client_found.client_id)

    db.update(op_found, status=OpStatus.COMPUTING.value)


@sio.on('result', namespace="/ravjs")
async def receive_result(sid, message):
    print("\nResult received...")
    print(message)
    data = json.loads(message)

    op_id = data['op_id']

    print(op_id, type(data['result']), data['operator'], data['result'])

    op = db.session.query(Op).get(op_id)
    data = data_manager.create_data(data=np.array(data['result']), data_type="ndarray")
    db.update(op, outputs=json.dumps([data.id]), status=OpStatus.COMPUTED.value)

    """
    Send a pending op
    """
    op_found = search_pending_op()

    if op_found is None:
        return

    # Create payload
    inputs = json.loads(op_found.inputs)

    payload = create_payload(op_found.id, inputs, op_found.op_type, op_found.operator)

    await sio.emit("op", payload, namespace="/ravjs", room=sid)

    db.update(op_found, client=sid, status=OpStatus.COMPUTING.value)

    # # Save results
    # # If compute completed
    # op_id = data.get("op_id", None)
    # if op_id is not None and data.get('result', None) is not None:
    #     print("Result dict:", data)
    #
    #     save_result(op_id, result=data['result'])
    #
    #     r.set(op_id + ":result", "Done")
    #
    #     r.lrem("ops_computing", op_id)
    #     r.rpush("ops_computed", op_id)
    #
    #     await sio.emit("result", {"op_id": op_id}, namespace="/ravop")
    #
    # # Send an op to this client if there is a pending op
    # pending_op = get_pending_op(r)
    # if pending_op is not None:
    #     await sio.emit("op", pending_op, namespace="/ravjs", room=sid)
    #
    #     r.lpop("ops_pending")
    #     r.rpush("ops_computing", pending_op['op_id'])


@sio.event
async def connect(sid, environ):
    print("Connected:", sid, environ)

    client_type = None
    if 'ravop' in environ['QUERY_STRING']:
        client_type = "ravop"
    elif 'ravjs' in environ['QUERY_STRING']:
        client_type = "ravjs"

    try:
        client = Client()
        client.client_id = sid
        client.connected_at = datetime.datetime.now()
        client.status = "connected"
        client.type = client_type
        db.session.add(client)
        db.session.commit()
    except:
        db.session.rollback()
        raise

    op_found = search_pending_op()

    if op_found is None:
        return

    # Create payload
    inputs = json.loads(op_found.inputs)

    payload = create_payload(op_found.id, inputs, op_found.op_type, op_found.operator)

    await sio.emit("op", payload, namespace="/ravjs", room=sid)

    db.update(op_found, client=sid, status=OpStatus.COMPUTING.value)


@sio.event
async def disconnect(sid):
    print("Disconnected:", sid)

    client = db.session.query(Client).filter(Client.client_id == sid).first()
    if client is not None:
        client.status = "disconnected"
        client.disconnected_at = datetime.datetime.now()


def search_pending_op():
    """
    Search for an op which is pending
    """
    ops = db.session.query(Op).filter(Op.status == "pending").filter(Op.client_id == None).all()

    print("Ops:", ops)
    op_found = None
    for op in ops:
        inputs = json.loads(op.inputs)

        not_computed = []
        for op_id in inputs:
            if db.session.query(Op).get(op_id).status != "computed":
                not_computed.append(op_id)

        if len(not_computed) == 0:
            op_found = op
            break

    return op_found


def search_client():
    clients = db.session.query(Client).filter(Client.status == "connected", Client.type == "ravjs")

    client_found = None
    for client in clients:
        op = db.session.query(Op).filter(Op.status == "computing", Op.client_id == client.id).first()
        if op is None:
            client_found = client
            break

    return client_found


def create_payload(op_id1, inputs, op_type, operator):
    """
    Create a payload
    """
    values = []

    for op_id in inputs:
        data_id = json.loads(db.session.query(Op).get(op_id).outputs)[0]
        print("Data id:", op_id, data_id)
        data = db.session.query(Data).get(data_id)
        file_path = data.file_path
        print(file_path)
        with open(file_path, "rb") as f:
            a = pickle.load(f)
            print("Data:", a, type(a), data.type)
            if data.type == "integer":
                values.append(a)
            elif data.type == "ndarray":
                print(type(a))
                values.append(a.tolist())
            else:
                print("Value:", a)
                values.append(a)

    payload = dict()
    payload['op_id'] = op_id1
    payload['values'] = values
    payload['op_type'] = op_type
    payload['operator'] = operator

    return json.dumps(payload)


# We bind our aiohttp endpoint to our app router
app.router.add_get('/', index)
