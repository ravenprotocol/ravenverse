import datetime
import json
import os
from enum import Enum

import numpy as np
import sqlalchemy as db
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

from common.utils import delete_data_file, save_data_to_file

Base = declarative_base()


class OpTypes(Enum):
    UNARY = "unary"
    BINARY = "binary"
    OTHER = "other"


class NodeTypes(Enum):
    INPUT = "input"
    MIDDLE = "middle"
    OUTPUT = "output"


class Operators(Enum):
    MATRIX_MULTIPLICATION = "matrix_multiplication"
    ADDITION = "addition"
    SUBTRACTION = "subtraction"
    MULTIPLICATION = "multiplication"
    ELEMENT_WISE_MULTIPLICATION = "element_wise_multiplication"
    DIVISION = "division"
    LINEAR = "linear"
    NEGATION = "negation"
    EXPONENTIAL = "exponential"
    TRANSPOSE = "transpose"
    NATURAL_LOG = "natural_log"
    MATRIX_SUM = "matrix_sum"
    SQUARE = "square"
    CUBE = "cube"
    SQUARE_ROOT = "square_root"
    CUBE_ROOT = "cube_root"


class OpStatus(Enum):
    PENDING = "pending"
    COMPUTING = "computing"
    COMPUTED = "computed"
    FAILED = "failed"


class Graph(Base):
    __tablename__ = 'graph'
    id = Column(Integer, primary_key=True)
    ops = relationship("Op", backref="graph")

    # Status of this graph 1.active 2.inactive
    status = Column(String(10), default="active")

    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class Data(Base):
    __tablename__ = 'data'
    id = Column(Integer, primary_key=True)
    type = Column(String(20), nullable=False)
    file_path = Column(String(200), nullable=True)
    value = Column(String(100), nullable=True)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class Client(Base):
    __tablename__ = 'client'
    id = Column(Integer, primary_key=True)
    client_id = Column(String(100), nullable=False)
    client_ip = Column(String(20), nullable=True)
    status = Column(String(20), nullable=False, default="disconnected")
    # 1. ravop 2. ravjs
    type = Column(String(10), nullable=True)

    connected_at = Column(DateTime, default=datetime.datetime.utcnow)
    disconnected_at = Column(DateTime, default=datetime.datetime.utcnow)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class Op(Base):
    __tablename__ = 'op'
    id = Column(Integer, primary_key=True)

    # Op name
    name = Column(String(20), nullable=True)

    # Graph id
    graph_id = Column(Integer, ForeignKey('graph.id'))

    # 1. input 2. output 3. middle
    node_type = Column(String(10), nullable=False)

    # Store list of op ids
    inputs = Column(Text, nullable=True)

    # Store filenames - Pickle files
    outputs = Column(String(100), nullable=True)

    # Op type for no change in values
    op_type = Column(String(50), nullable=False)
    operator = Column(String(50), nullable=False)

    # 1. pending 2. computing 3. computed
    status = Column(String(10), default="pending")

    # List of clients who are working on it
    client_id = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class Singleton:

    def __init__(self, cls):
        self._cls = cls

    def Instance(self):
        try:
            return self._instance
        except AttributeError:
            self._instance = self._cls()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `Instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._cls)


@Singleton
class DBManager(object):
    def __init__(self):
        self.engine = db.create_engine('mysql://{}:{}@localhost/ravenwebdemo'.format(
            os.environ.get('MYSQL_USER'),
            os.environ.get('MYSQL_PASSWORD')), isolation_level="READ UNCOMMITTED")
        self.connection = self.engine.connect()

        Base.metadata.bind = self.engine

    def create_session(self):
        """
        Create a new session
        """
        DBSession = sessionmaker(bind=self.engine)
        return DBSession()

    def create_tables(self):
        Base.metadata.create_all(self.engine)

    def refresh(self, session, obj):
        """
        Refresh an object
        """
        session.refresh(obj)

    def get(self, name, id):
        session = self.create_session()

        if name == "op":
            obj = session.query(Op).get(id)
        elif name == "data":
            obj = session.query(Data).get(id)
        elif name == "graph":
            obj = session.query(Graph).get(id)
        elif name == "client":
            obj = session.query(Client).get(id)
        else:
            obj = None

        if obj is None:
            session.close()
            return None

        obj_dict = obj.__dict__
        session.close()
        return obj_dict

    def add(self, name, **kwargs):
        session = self.create_session()

        if name == "op":
            obj = Op()
        elif name == "data":
            obj = Data()
        elif name == "graph":
            obj = Graph()
        elif name == "client":
            obj = Client()
        else:
            obj = None

        for key, value in kwargs.items():
            setattr(obj, key, value)
        session.add(obj)
        session.commit()

        id = obj.id
        session.close()
        return id

    def update(self, name, id, **kwargs):
        session = self.create_session()

        if name == "op":
            obj = session.query(Op).get(id)
        elif name == "data":
            obj = session.query(Data).get(id)
        elif name == "graph":
            obj = session.query(Graph).get(id)
        elif name == "client":
            obj = session.query(Client).get(id)
        else:
            obj = None

        if obj is not None:
            for key, value in kwargs.items():
                setattr(obj, key, value)
            session.commit()
        else:
            raise Exception("Invalid id")
        session.close()

    def delete(self, name, id):
        session = self.create_session()

        if name == "op":
            obj = session.query(Op).get(id)
        elif name == "data":
            obj = session.query(Data).get(id)
        elif name == "graph":
            obj = session.query(Graph).get(id)
        elif name == "client":
            obj = session.query(Client).get(id)
        else:
            obj = None

        if obj is not None:
            session.delete(obj)
            session.commit()

        session.close()

    def create_op(self, **kwargs):
        session = self.create_session()
        op = Op()

        for key, value in kwargs.items():
            setattr(op, key, value)

        session.add(op)
        session.commit()
        id = op.id
        session.close()
        return id

    def get_op(self, op_id):
        """
        Get an existing op
        """
        session = self.create_session()
        op = session.query(Op).get(op_id)
        if op is None:
            session.close()
            return None

        op_dict = op.__dict__
        session.close()
        return op_dict

    def update_op(self, op_id, **kwargs):
        session = self.create_session()
        obj = session.query(Op).get(op_id)

        for key, value in kwargs.items():
            setattr(obj, key, value)

        session.commit()
        session.close()

    def create_data(self, **kwargs):
        session = self.create_session()
        data = Data()

        for key, value in kwargs.items():
            setattr(data, key, value)

        session.add(data)
        session.commit()
        id = data.id
        session.close()
        return id

    def get_data(self, data_id):
        """
        Get an existing data
        """
        session = self.create_session()
        data = session.query(Data).get(data_id)
        data_dict = data.__dict__
        session.close()
        return data_dict

    def update_data(self, data_id, **kwargs):
        session = self.create_session()
        obj = session.query(Data).get(data_id)

        for key, value in kwargs.items():
            setattr(obj, key, value)

        session.commit()
        session.close()

    def delete_data(self, data_id):
        session = self.create_session()
        data = session.query(Data).get(data_id)
        session.delete(data)
        session.commit()
        session.close()

    def create_data_complete(self, data, data_type):
        # print("Creating data:", data)

        if isinstance(data, (np.ndarray, np.generic)):
            if data.ndim == 1:
                data = data[..., np.newaxis]

        d = self.create_data(type=data_type)

        # Save file
        file_path = save_data_to_file(d.id, data, data_type)

        # Update file path
        self.update(d, file_path=file_path)

        return d

    def get_op_status(self, op_id):
        session = self.create_session()
        status = session.query(Op).get(op_id).status
        session.close()
        return status

    def get_graph(self, graph_id):
        """
        Get an existing graph
        """
        session = self.create_session()
        graph = session.query(Graph).get(graph_id)
        graph_dict = graph.__dict__
        session.close()
        return graph_dict

    def create_graph(self):
        """
        Create a new graph
        """
        session = self.create_session()
        graph = Graph()
        session.add(graph)
        session.commit()
        id = graph.id
        session.close()
        return id

    def get_graph_ops(self, graph_id):
        session = self.create_session()
        ops = session.query(Op).filter(Op.graph_id == graph_id).all()
        session.close()
        return ops

    def delete_graph_ops(self, graph_id):
        print("Deleting graph ops")
        session = self.create_session()
        ops = self.get_graph_ops(graph_id=graph_id)

        for op in ops:
            print("Op id:{}".format(op.id))
            data_ids = json.loads(op.outputs)
            if data_ids is not None:
                for data_id in data_ids:
                    print("Data id:{}".format(data_id))

                    # Delete data file
                    delete_data_file(data_id)

                    # Delete data object
                    self.delete_data(data_id)

            # Delete op object
            self.delete(op)

        session.close()

    def create_client(self, **kwargs):
        session = self.create_session()
        obj = Client()

        for key, value in kwargs.items():
            setattr(obj, key, value)

        session.add(obj)
        session.commit()
        return obj

    def get_client(self, client_id):
        """
        Get an existing data
        """
        session = self.create_session()
        obj = session.query(Client).get(client_id)
        session.close()
        return obj

    def update_client(self, client_id, **kwargs):
        session = self.create_session()
        obj = session.query(Client).get(client_id)

        for key, value in kwargs.items():
            setattr(obj, key, value)

        session.commit()
        session.close()
