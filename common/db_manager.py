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
    ELEMENT_WISE_MULTIPLICATION = "element_wise_multiplication"
    DIVISION = "division"
    LINEAR = "linear"
    NEGATION = "negation"
    EXPONENTIAL = "exponential"
    TRANSPOSE = "transpose"
    NATURAL_LOG = "natural_log"


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
    file_path = Column(String(50), nullable=True)

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


class DBManager(object):
    def __init__(self):
        self.engine = db.create_engine('sqlite:///raven_db.db')
        self.connection = self.engine.connect()

        Base.metadata.bind = self.engine

        DBSession = sessionmaker(bind=self.engine)

        self.session = DBSession()

    def create_tables(self):
        Base.metadata.create_all(self.engine)

    def add(self, obj):
        self.session.add(obj)
        self.session.commit()
        return obj

    def update(self, obj, **kwargs):
        for key, value in kwargs.items():
            setattr(obj, key, value)

        self.session.commit()

    def get_session(self):
        return self.session

    def delete(self, obj):
        self.session.delete(obj)
        self.session.commit()

    def create_op(self, **kwargs):
        op = Op()

        for key, value in kwargs.items():
            setattr(op, key, value)

        self.session.add(op)
        self.session.commit()
        return op

    def get_op(self, op_id):
        """
        Get an existing op
        """
        return self.session.query(Op).get(op_id)

    def create_data(self, **kwargs):
        data = Data()

        for key, value in kwargs.items():
            setattr(data, key, value)

        self.session.add(data)
        self.session.commit()
        return data

    def get_data(self, data_id):
        """
        Get an existing data
        """
        return self.session.query(Data).get(data_id)

    def create_data_complete(self, data, data_type):
        # print("Creating data:", data)

        if isinstance(data, (np.ndarray, np.generic)):
            if data.ndim == 1:
                data = data[..., np.newaxis]

        d = self.create_data(type=data_type)

        # Save file
        file_path = save_data_to_file(d.id, data)

        # Update file path
        self.update(d, file_path=file_path)

        return d

    def get_op_status(self, op_id):
        return self.session.query(Op).get(op_id).status

    def get_graph(self, graph_id):
        """
        Get an existing graph
        """
        return self.session.query(Graph).get(graph_id)

    def create_graph(self):
        """
        Create a new graph
        """
        graph = Graph()
        self.session.add(graph)
        self.session.commit()
        return graph

    def get_graph_ops(self, graph_id):
        return self.session.query(Op).filter(Op.graph_id == graph_id).all()

    def delete_graph_ops(self, graph_id):
        ops = self.get_graph_ops(graph_id=graph_id)

        for op in ops:
            data_ids = json.loads(op.outputs)
            if data_ids is not None:
                for data_id in data_ids:
                    data = self.session.query(Data).get(data_id)
                    # Delete data file
                    delete_data_file(data)
                    # Delete data object
                    self.delete(data)

            # Delete op object
            self.delete(op)

