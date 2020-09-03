import datetime
import json
import logging
import logging.handlers
import pickle

import numpy as np
import socketio
from aiohttp import web

# creates a new Async Socket IO Server
from sqlalchemy import desc

from common import db

from common.db_manager import Client, Op, Data, OpStatus, Graph
from .constants import RAVSOCK_LOG_FILE
from ravop.core import Op as RavOp, Data as RavData

# Set up a specific logger with our desired output level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add the log message handler to the logger
handler = logging.handlers.RotatingFileHandler(RAVSOCK_LOG_FILE)

logger.addHandler(handler)

sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode='aiohttp', async_handlers=True)

# Creates a new Aiohttp Web Application
app = web.Application()

# Binds our Socket.IO server to our Web App instance
sio.attach(app)


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

    if op_found is None or client_found is None:
        return

    print("Op:", op_found.id, )
    print("Client:", client_found.id, client_found.client_id)

    # Create payload
    inputs = json.loads(op_found.inputs)

    payload = create_payload(op_found.id, inputs, op_found.op_type, op_found.operator)

    await sio.emit("op", payload, namespace="/ravjs", room=client_found.client_id)

    # db.update(op_found, client_id=client_found.client_id, status=OpStatus.COMPUTING.value)


@sio.on('acknowledge', namespace="/ravjs")
async def acknowledge(sid, message):
    print("Op received", sid)

    data = json.loads(message)
    op_id = data['op_id']
    print("Op id", op_id)
    session = db.create_session()
    op_found = session.query(Op).filter(Op.id == op_id).filter(Op.client_id == sid).first()

    # Update op status to computing
    if op_found is not None:
        print("Updating op status")
        db.update_op(op_found, client_id=sid, status=OpStatus.COMPUTING.value)

    session.close()


@sio.on('result', namespace="/ravjs")
async def receive_result(sid, message):
    print("\nResult received...")
    print(message)
    data = json.loads(message)

    op_id = data['op_id']

    print(op_id, type(data['result']), data['operator'], data['result'])

    op = RavOp(id=op_id)
    data = RavData(value=np.array(data['result']), dtype="ndarray")

    # op = db.session.query(Op).get(op_id)
    # data = db.create_data_complete(data=np.array(data['result']), data_type="ndarray")
    print(json.dumps([data.id]))
    db.update_op(op._op_db, outputs=json.dumps([data.id]), status=OpStatus.COMPUTED.value)

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

    # db.update(op_found, client_id=sid, status=OpStatus.COMPUTING.value)

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
    logger.debug("Connected:{} {}".format(sid, environ))

    client_type = None
    if 'ravop' in environ['QUERY_STRING']:
        client_type = "ravop"
    elif 'ravjs' in environ['QUERY_STRING']:
        client_type = "ravjs"

    session = db.create_session()

    try:
        client = Client()
        client.client_id = sid
        client.connected_at = datetime.datetime.now()
        client.status = "connected"
        client.type = client_type
        session.add(client)
        session.commit()
    except:
        session.rollback()
        session.close()
        raise


@sio.on('ask_op', namespace="/ravjs")
async def ask_op(sid, message):
    print("get_op", message)

    op_found = search_pending_op()
    client_found = search_client()

    if op_found is None or client_found is None:
        print("Op or client not found")
        return

    # Create payload
    inputs = json.loads(op_found.inputs)

    payload = create_payload(op_found.id, inputs, op_found.op_type, op_found.operator)

    print("Emitting op")
    print("sid", client_found.client_id, payload)

    await sio.emit("op", payload, namespace="/ravjs", room=client_found.client_id)


@sio.event
async def disconnect(sid):
    logger.debug("Disconnected:{}".format(sid))

    session = db.create_session()
    client = session.query(Client).filter(Client.client_id == sid).first()
    if client is not None:
        db.update_client(client, status="disconnected", disconnected_at = datetime.datetime.now())

        if client.type == "ravjs":
            # Get ops which were assigned to this
            ops = session.query(Op).filter(Op.status == "computing").filter(Op.client_id == sid).all()

            # Set those ops to pending
            for op in ops:
                db.update_op(op, client_id=None, status=OpStatus.PENDING.value)

    session.close()


def search_pending_op():
    """
    Search for an op which is pending
    """
    graphs = db.session.query(Graph).filter(Graph.status=="active")
    graph_id = None
    for graph in graphs:
        graph_id = graph.id
        break

    if graph_id is not None:
        ops = db.session.query(Op).filter(Op.graph_id==graph_id).filter(Op.status == "pending").filter(Op.client_id == None)
        #.order_by(desc(Op.created_at))

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
    return None


def search_client():
    session = db.create_session()
    clients = session.query(Client).filter(Client.status == "connected", Client.type == "ravjs").order_by(desc(Client.created_at))

    client_found = None
    for client in clients:
        op = session.query(Op).filter(Op.status == "computing", Op.client_id == client.id).first()
        if op is None:
            client_found = client
            break

    session.close()
    return client_found


def create_payload(op_id1, inputs, op_type, operator):
    """
    Create a payload
    """
    values = []

    for op_id in inputs:
        op = RavOp(id=op_id)
        if op.output_dtype == "ndarray":
            values.append(op.output.tolist())
        else:
            values.append(op.output)

        # data_id = json.loads(db.session.query(Op).get(op_id).outputs)[0]
        # print("Data id:", op_id, data_id)
        # data = db.session.query(Data).get(data_id)
        # file_path = data.file_path
        # print(file_path)
        #
        # if data.type == "ndarray":
        #
        #
        # with open(file_path, "rb") as f:
        #     a = json.load(f)
        #     print("Data:", a, type(a), data.type)
        #     if data.type == "integer":
        #         values.append(a)
        #     elif data.type == "ndarray":
        #         a = np.array(a)
        #         print(type(a))
        #         values.append(a.tolist())
        #     else:
        #         print("Value:", a)
        #         values.append(a)

    payload = dict()
    payload['op_id'] = op_id1
    payload['values'] = values
    payload['op_type'] = op_type
    payload['operator'] = operator

    return json.dumps(payload)


async def schedule_ops():
    print("ehlo")


# We bind our aiohttp endpoint to our app router
app.router.add_get('/', index)
