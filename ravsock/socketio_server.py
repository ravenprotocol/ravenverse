import datetime
import json
import logging
import logging.handlers
import numpy as np
import socketio
from aiohttp import web

# creates a new Async Socket IO Server
from sqlalchemy import desc, and_, or_

from common import db

from common.db_manager import Client, Op, Data, OpStatus, Graph, ClientOpMapping, ClientOpMappingStatus
from common import RavQueue
from .constants import RAVSOCK_LOG_FILE, QUEUE_HIGH_PRIORITY, QUEUE_LOW_PRIORITY, QUEUE_COMPUTING
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

# Instantiate queues
queue_high_priority = RavQueue(name=QUEUE_HIGH_PRIORITY)
queue_low_priority = RavQueue(name=QUEUE_LOW_PRIORITY)
queue_computing = RavQueue(name=QUEUE_COMPUTING)


# we can define aiohttp endpoints just as we normally
# would with no change
async def index(request):
    with open('../ravclient/index.html') as f:
        return web.Response(text=f.read(), content_type='text/html')


"""
Connect and disconnect events
"""


@sio.event
async def connect(sid, environ):
    logger.debug("Connected:{} {}".format(sid, environ))

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
    except Exception as e:
        logger.debug(str(e))
        db.session.rollback()
        raise


@sio.event
async def disconnect(sid):
    logger.debug("Disconnected:{}".format(sid))

    client = db.get_client_by_sid(sid=sid)
    if client is not None:
        db.update_client(client, status="disconnected", disconnected_at=datetime.datetime.now())

        if client.type == "ravjs":
            # Get ops which were assigned to this
            ops = db.session.query(ClientOpMapping).filter(ClientOpMapping.client_id ==
                                                           sid).filter(or_(ClientOpMapping.status
                                                                           == ClientOpMappingStatus.SENT.value,
                                                                           ClientOpMapping.status ==
                                                                           ClientOpMappingStatus.ACKNOWLEDGED,
                                                                           ClientOpMapping.status ==
                                                                           ClientOpMappingStatus.COMPUTING.value)).all()

            print(ops)
            # Set those ops to pending
            for op in ops:
                db.update_op(op, status=ClientOpMappingStatus.NOT_COMPUTED.value)


"""
Ping and pong events
"""


@sio.on("ping", namespace="/ravjs")
async def ping(sid):
    await sio.emit("pong", {}, namespace="/ravjs", room=sid)


@sio.on("pong", namespace="/ravjs")
async def pong(sid):
    """
    Client is available
    Send an op to the client
    """
    print("Pong: {}".format(sid))

    # Find, create payload and emit op
    await emit_op(sid)


@sio.on('inform_server', namespace="/ravop")
async def inform_server(sid, data):
    print("Inform server")
    data_type = data['type']
    if data_type == "op":
        data_id = data['op_id']

        # Emit op to the client
        client = db.get_available_clients()[0]
        await emit_op(client.client_id, data_id)
    else:
        # Emit op to the client
        clients = db.get_available_clients()[:3]
        print(clients)
        for client in clients:
            await sio.emit("ping", data=None, namespace="/ravjs", room=client.client_id)


@sio.on('remind_server', namespace="/ravop")
async def remind_server(sid, data):
    data = json.load(data)
    data_type = data['type']
    if data_type == "op":
        data_id = data['op_id']
    else:
        data_id = data['graph_id']


"""
When clients asks for an op
1. Op Computed or failed
"""


@sio.on('get_op', namespace="/ravjs")
async def get_op(sid, message):
    """
    Send an op to the client
    """
    print("get_op", message)

    # Find, create payload and emit op
    await emit_op(sid)


@sio.on('acknowledge_op', namespace="/ravjs")
async def acknowledge_op(sid, message):
    print("Op received", sid)

    data = json.loads(message)
    op_id = data['op_id']
    print("Op id", op_id)
    op_found = db.get_op(op_id=op_id)

    if op_found is not None:
        # Update client op mapping - Status to acknowledged
        update_client_op_mapping(op_id, sid, ClientOpMappingStatus.ACKNOWLEDGED.value)


@sio.on('op_completed', namespace="/ravjs")
async def op_completed(sid, data):
    # Save the results
    print("\nResult received {}".format(data))
    data = json.loads(data)

    op_id = data['op_id']

    print(op_id, type(data['result']), data['operator'], data['result'])

    op = RavOp(id=op_id)

    if data["status"] == "success":
        data = RavData(value=np.array(data['result']), dtype="ndarray")

        # Update op
        db.update_op(op._op_db, outputs=json.dumps([data.id]), status=OpStatus.COMPUTED.value)

        # Update client op mapping
        update_client_op_mapping(op_id, sid=sid, status=ClientOpMappingStatus.COMPUTED.value)
    else:
        # Update op
        db.update_op(op._op_db, outputs=None, status=OpStatus.FAILED.value)

        # Update client op mapping
        update_client_op_mapping(op_id, sid=sid, status=ClientOpMappingStatus.FAILED.value)

    # Emit another op to this client
    await emit_op(sid)


"""
1. Find Op
2. Create Payload
3. Emit Op
"""


async def emit_op(sid, op=None):
    """
    1. Find an op
    2. Create payload
    3. Emit Op
    """
    # Find an op
    if op is None:
        op = find_op()

    if op is None:
        return

    # Create payload
    payload = create_payload(op)

    # Emit op
    print("Emitting op:{}, {}".format(sid, payload))
    await sio.emit("op", payload, namespace="/ravjs", room=sid)

    # Store the mapping in database
    client = db.get_client_by_sid(sid)
    ravop = RavOp(id=op.id)
    db.update_op(ravop._op_db, status=OpStatus.COMPUTING.value)
    mapping = db.create_client_op_mapping(client_id=client.id, op_id=op.id, sent_time=datetime.datetime.now(),
                                          status=ClientOpMappingStatus.SENT.value)
    logger.debug("Mapping created:{}".format(mapping))


def find_op():
    """
    Find op
    1. Look in mappings
    2. Pop from the queues
    """

    op = db.get_incomplete_op()

    if op is not None:
        return op
    else:
        q1 = RavQueue(name=QUEUE_HIGH_PRIORITY)
        q2 = RavQueue(name=QUEUE_LOW_PRIORITY)

        while True:
            if q1.__len__() > 0:
                op_id = q1.pop()
                op = db.get_op(op_id=op_id)
                return op
            elif q2.__len__() > 0:
                op_id = q2.pop()
                op = db.get_op(op_id=op_id)
                return op
            else:
                print("There is no op")
                return None


def create_payload(op):
    """
    Create payload for the operation
    params:
    op: database op
    """
    values = []
    inputs = json.loads(op.inputs)
    for op_id in inputs:
        ravop = RavOp(id=op_id)
        if ravop.output_dtype == "ndarray":
            values.append(ravop.output.tolist())
        else:
            values.append(ravop.output)

    payload = dict()
    payload['op_id'] = op.id
    payload['values'] = values
    payload['op_type'] = op.op_type
    payload['operator'] = op.operator

    return json.dumps(payload)


def update_client_op_mapping(op_id, sid, status):
    client = db.get_client_by_sid(sid)
    mapping = db.find_client_op_mapping(client.id, op_id)
    db.update_client_op_mapping(mapping.id, status=status,
                                response_time=datetime.datetime.now())

#
# @sio.on('update_server', namespace="/ravop")
# async def receive_op(sid):
#     print("\nOp Received...", sid)
#
#     client_found = search_client()
#     op_found = search_pending_op()
#
#     if op_found is None or client_found is None:
#         return
#
#     print("Op:", op_found.id, )
#     print("Client:", client_found.id, client_found.client_id)
#
#     # Create payload
#     inputs = json.loads(op_found.inputs)
#
#     payload = create_payload(op_found.id, inputs, op_found.op_type, op_found.operator)
#
#     await sio.emit("op", payload, namespace="/ravjs", room=client_found.client_id)
#
#     # db.update(op_found, client_id=client_found.client_id, status=OpStatus.COMPUTING.value)
#
#
# # @sio.on('result', namespace="/ravjs")
# # async def receive_result(sid, message):
# #     print("\nResult received...")
# #     print(message)
# #     data = json.loads(message)
# #
# #     op_id = data['op_id']
# #
# #     print(op_id, type(data['result']), data['operator'], data['result'])
# #
# #     op = RavOp(id=op_id)
# #     data = RavData(value=np.array(data['result']), dtype="ndarray")
# #
# #     # op = db.session.query(Op).get(op_id)
# #     # data = db.create_data_complete(data=np.array(data['result']), data_type="ndarray")
# #     print(json.dumps([data.id]))
# #     db.update_op(op._op_db, outputs=json.dumps([data.id]), status=OpStatus.COMPUTED.value)
# #
# #     """
# #     Send a pending op
# #     """
# #     op_found = search_pending_op()
# #
# #     if op_found is None:
# #         return
# #
# #     # Create payload
# #     inputs = json.loads(op_found.inputs)
# #
# #     payload = create_payload(op_found.id, inputs, op_found.op_type, op_found.operator)
# #
# #     await sio.emit("op", payload, namespace="/ravjs", room=sid)
# #
# #     # db.update(op_found, client_id=sid, status=OpStatus.COMPUTING.value)
# #
# #     # # Save results
# #     # # If compute completed
# #     # op_id = data.get("op_id", None)
# #     # if op_id is not None and data.get('result', None) is not None:
# #     #     print("Result dict:", data)
# #     #
# #     #     save_result(op_id, result=data['result'])
# #     #
# #     #     r.set(op_id + ":result", "Done")
# #     #
# #     #     r.lrem("ops_computing", op_id)
# #     #     r.rpush("ops_computed", op_id)
# #     #
# #     #     await sio.emit("result", {"op_id": op_id}, namespace="/ravop")
# #     #
# #     # # Send an op to this client if there is a pending op
# #     # pending_op = get_pending_op(r)
# #     # if pending_op is not None:
# #     #     await sio.emit("op", pending_op, namespace="/ravjs", room=sid)
# #     #
# #     #     r.lpop("ops_pending")
# #     #     r.rpush("ops_computing", pending_op['op_id'])
#
#
# # async def second_c(*data):
# #     print(data)
#
#
# # @sio.on("callback", namespace="/ravop")
# # async def my_callback(sid, data):
# #     print(sid)
# #     await sio.emit("your", {}, room=sid, callback=second_c)
# #     return ["received"]
#
#
# @sio.on('ask_op', namespace="/ravjs")
# async def ask_op(sid, message):
#     print("get_op", message)
#
#     op_found = search_pending_op()
#     client_found = search_client()
#
#     if op_found is None or client_found is None:
#         print("Op or client not found")
#         return
#
#     # Create payload
#     inputs = json.loads(op_found.inputs)
#
#     payload = create_payload(op_found.id, inputs, op_found.op_type, op_found.operator)
#
#     print("Emitting op")
#     print("sid", client_found.client_id, payload)
#
#     await sio.emit("op", payload, namespace="/ravjs", room=client_found.client_id)
#
#
# def search_pending_op():
#     """
#     Search for an op which is pending
#     """
#     graphs = db.session.query(Graph).filter(Graph.status == "active")
#     graph_id = None
#     for graph in graphs:
#         graph_id = graph.id
#         break
#
#     if graph_id is not None:
#         ops = db.session.query(Op).filter(Op.graph_id == graph_id).filter(Op.status == "pending").filter(
#             Op.client_id is None)
#         # .order_by(desc(Op.created_at))
#
#         print("Ops:", ops)
#         op_found = None
#         for op in ops:
#             inputs = json.loads(op.inputs)
#
#             not_computed = []
#             for op_id in inputs:
#                 if db.session.query(Op).get(op_id).status != "computed":
#                     not_computed.append(op_id)
#
#             if len(not_computed) == 0:
#                 op_found = op
#                 break
#     return op_found
#     # return None
#
#
# def search_client():
#     clients = db.session.query(Client).filter(Client.status == "connected", Client.type == "ravjs").order_by(
#         desc(Client.created_at))
#
#     client_found = None
#     for client in clients:
#         op = db.session.query(Op).filter(Op.status == "computing", Op.client_id == client.id).first()
#         if op is None:
#             client_found = client
#             break
#
#     return client_found
#
#
# def create_payload(op_id1, inputs, op_type, operator):
#     """
#     Create a payload
#     """
#     values = []
#
#     for op_id in inputs:
#         op = RavOp(id=op_id)
#         if op.output_dtype == "ndarray":
#             values.append(op.output.tolist())
#         else:
#             values.append(op.output)
#
#         # data_id = json.loads(db.session.query(Op).get(op_id).outputs)[0]
#         #         # print("Data id:", op_id, data_id)
#         #         # data = db.session.query(Data).get(data_id)
#         #         # file_path = data.file_path
#         #         # print(file_path)
#         #
#         # if data.type == "ndarray":
#         #
#         #
#         # with open(file_path, "rb") as f:
#         #     a = json.load(f)
#         #     print("Data:", a, type(a), data.type)
#         #     if data.type == "integer":
#         #         values.append(a)
#         #     elif data.type == "ndarray":
#         #         a = np.array(a)
#         #         print(type(a))
#         #         values.append(a.tolist())
#         #     else:
#         #         print("Value:", a)
#         #         values.append(a)
#
#     payload = dict()
#     payload['op_id'] = op_id1
#     payload['values'] = values
#     payload['op_type'] = op_type
#     payload['operator'] = operator
#
#     return json.dumps(payload)


# We bind our aiohttp endpoint to our app router
app.router.add_get('/', index)
