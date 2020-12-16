from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy

from common import RavQueue
from common.constants import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
from ravop.core import QUEUE_HIGH_PRIORITY, QUEUE_LOW_PRIORITY, Graph
from common import db as db_manager


app = Flask(__name__)
app.config["DEBUG"] = True
SQLALCHEMY_DATABASE_URI = 'mysql://{}:{}@{}:{}/{}?host={}?port={}'.format(MYSQL_USER, MYSQL_PASSWORD,MYSQL_HOST,
                                                                          MYSQL_PORT, MYSQL_DATABASE,MYSQL_HOST,
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
        ops_list.append(op.__dict__)
    return render_template('ops.html', ops=ops_list)


if __name__ == '__main__':
    app.run()
