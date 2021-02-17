import ast
import json
import datetime

from flask import Flask, render_template, jsonify, Response
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect

from common import RavQueue
from common.constants import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
from ravop.core import QUEUE_HIGH_PRIORITY, QUEUE_LOW_PRIORITY, Graph, Op
from common import db as db_manager

app = Flask(__name__, static_folder='static')
app.config["DEBUG"] = True
SQLALCHEMY_DATABASE_URI = 'mysql://{}:{}@{}:{}/{}?host={}?port={}'.format(MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST,
                                                                          MYSQL_PORT, MYSQL_DATABASE, MYSQL_HOST,
                                                                          MYSQL_PORT)
app.config['SQLALCHEMY_DATABASE_URI'] = SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

queue_high_priority = RavQueue(name=QUEUE_HIGH_PRIORITY)
queue_low_priority = RavQueue(name=QUEUE_LOW_PRIORITY)


def get_queued_ops():
    high_priority_ops = []
    for i in range(queue_high_priority.__len__()):
        high_priority_ops.append(queue_high_priority.get(i))

    low_priority_ops = []
    for i in range(queue_low_priority.__len__()):
        low_priority_ops.append(queue_low_priority.get(i))

    return high_priority_ops, low_priority_ops


@app.route('/')
def home():
    high_priority_ops, low_priority_ops = get_queued_ops()
    return render_template('home.html', high_priority_ops=high_priority_ops, low_priority_ops=low_priority_ops)


@app.route('/clients')
def clients():
    clients = db_manager.get_all_clients()
    clients_list = []
    for client in clients:
        clients_list.append(client.__dict__)
    return render_template('clients.html', clients=clients_list)


@app.route('/graphs')
def graphs():
    graphs = db_manager.get_all_graphs()
    graphs_list = []
    for graph in graphs:
        progress = Graph(id=graph.id).progress
        graph_dict = graph.__dict__
        graph_dict['progress'] = progress
        graphs_list.append(graph_dict)
    return render_template('graphs.html', graphs=graphs_list)


@app.route('/ops')
def ops():
    ops = db_manager.get_all_ops()
    ops_list = []
    for op in ops:
        op_dict = op.__dict__
        op_dict = parse_op_inputs_outputs(op_dict)
        ops_list.append(op_dict)
    return render_template('ops.html', ops=ops_list)


@app.route('/graph/viz/<graph_id>/')
def graph_vis(graph_id):
    graph = db_manager.get_graph(graph_id=graph_id)
    ops = db_manager.get_ops(graph_id=graph_id)
    ops_list = []
    for op in ops:
        op_dict = object_as_dict(op)
        op_dict = parse_op_inputs_outputs(op_dict)
        # op_dict = datetime_encoder(op_dict)
        ops_list.append(op_dict)
    return render_template('graph_viz.html', graph=object_as_dict(graph), ops=ops_list)


@app.route('/graph/ops/<graph_id>/json/', methods=['GET'])
def graph_ops_json(graph_id):
    ops = db_manager.get_ops(graph_id=graph_id)
    ops_list = []
    for op in ops:
        # op = op.serialize()
        print(op)
        op_dict = object_as_dict(op)
        # op_dict = parse_op_inputs_outputs(op_dict)
        op_dict = datetime_encoder(op_dict)
        # print(op_dict)
        ops_list.append(op_dict)
    return jsonify(ops_list)


@app.route('/graph/ops/<graph_id>/')
def graph_ops(graph_id):
    ops = db_manager.get_ops(graph_id=graph_id)
    ops_list = []
    for op in ops:
        op_dict = op.__dict__
        op_dict = parse_op_inputs_outputs(op_dict)
        ops_list.append(op_dict)
    return render_template('graph_ops.html', ops=ops_list)


@app.route('/graph/ops/<graph_id>/<graph_op_id>/')
def graph_op_viewer(graph_id, graph_op_id):
    op = db_manager.get_op(op_id=graph_op_id)
    op_dict = op.__dict__
    op_dict = parse_op_inputs_outputs(op_dict)
    return render_template('op_viewer.html', op=op_dict)


@app.route('/data/<data_id>/')
def data_viewer(data_id):
    data = db_manager.get_data(data_id=data_id)
    data_dict = data.__dict__
    op = Op(id=data_dict["id"])
    print(type(op.output).__name__)
    if type(op.output).__name__ == 'float' or type(op.output).__name__ == "int":
        data_dict['output'] = op.output
        data_dict['shape'] = 1
    else:
        data_dict['output'] = op.output.tolist()
        data_dict['shape'] = op.output.shape
    return render_template('data_viewer.html', data=data_dict)


@app.route('/graph/<graph_id>/')
def graph_viewer(graph_id):
    graph = db_manager.get_graph(graph_id=graph_id)
    return render_template('graph_viewer.html', graph=graph.__dict__)


def parse_op_inputs_outputs(op_dict):
    # print(op_dict['inputs'])
    if op_dict.get("inputs") is not None and op_dict.get("inputs") != "null":
        # print(op_dict)
        inputs = ast.literal_eval(op_dict['inputs'])
        inputs_list = []
        for op_id in inputs:
            op_dict = object_as_dict(db_manager.get_op(op_id=op_id))
            # op_dict = datetime_encoder(op_dict)
            inputs_list.append(op_dict)
        op_dict['inputs'] = inputs_list
    else:
        op_dict["inputs"] = []

    if op_dict.get("outputs") is not None and op_dict.get("outputs") != "null":
        outputs = ast.literal_eval(op_dict['outputs'])
        outputs_list = []
        for data_id in outputs:
            data_dict = object_as_dict(db_manager.get_data(data_id=data_id))
            # data_dict = datetime_encoder(data_dict)
            outputs_list.append(data_dict)
        op_dict['outputs'] = outputs_list
    else:
        op_dict['outputs'] = []

    return op_dict


def object_as_dict(obj):
    return {c.key: getattr(obj, c.key)
            for c in inspect(obj).mapper.column_attrs}


def datetime_encoder(obj):
    result = {}
    for key, value in obj.items():
        if isinstance(value, (datetime.date, datetime.datetime)):
            print(key, value)
            result[key] = str(value)
        else:
            result[key] = value

    return result


if __name__ == '__main__':
    app.run()
