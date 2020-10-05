import os
import re
import time

from sklearn.datasets import load_breast_cancer, load_boston
import sklearn
from django.conf import settings

import app.models as models
import app.constants as constants
from ravop.socket_client import SocketClient
from app.utils import computations

from django.core.exceptions import ObjectDoesNotExist
from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import render

from ravop.core import Op, Scalar, Tensor
from ravop.ml import LogisticRegression, LinearRegression

import json
import numpy as np
import pandas as pd


operators = [
    {
        "name": "Addition", "type": "binary", "func": "add"
    },
    {
        "name": "Subtraction", "type": "binary", "func": "sub"
    },
    {
        "name": "Matrix Multiplication", "type": "binary", "func": "matmul"
    },
    {
        "name": "Multiplication", "type": "binary", "func": "elemul"
    },
    {
        "name": "Division", "type": "binary", "func": "div"
    },
    {
        "name": "Element-wise Multiplication", "type": "binary", "func": "elemul"
    },
    {
        "name": "Negation", "type": "unary", "func": "neg"
    },
    {
        "name": "Exponential", "type": "unary", "func": "exp"
    },
    {
        "name": "Transpose", "type": "unary", "func": "trans"
    },
    {
        "name": "Natural Log", "type": "unary", "func": "natlog"
    },
    {
        "name": "Matrix Sum", "type": "unary", "func": "matsum"
    },
    {
        "name": "Linear", "type": "unary", "func": "linear"
    }
]


def home(request):
    return render(request, "home.html")


def get_operator_name(operator_number):
    operators = [e.value for e in constants.Operators]
    return operators[operator_number - 1]


# def parse_data(data):
#     if not data:
#         return data, None
#     try:
#         return int(data), 'integer'
#     except ValueError:
#         try:
#             return float(data), 'double'
#         except ValueError:
#             try:
#                 data = json.loads(data)
#                 data = np.array(data)
#                 return data, 'ndarray'
#             except ValueError:
#                 return 'invalid_data', ''


def parse_data(data):
    if type(data).__name__ == "str":
        data = json.loads(data)
        if type(data).__name__ == "list":
            return Tensor(data)
        elif type(data).__name__ in ["int", "float", "double"]:
            return Scalar(data)
    elif type(data).__name__ == "list":
        return Tensor(data)
    elif type(data).__name__ in ["int", "float", "double"]:
        return Scalar(data)
    else:
        return None


class Compute(APIView):
    def post(self, request):
        print(request.data)
        operation = request.data.get('operation', None)

        if operation is None:
            return Response(data='Required parameters missing', status=400)

        operator = next(item for item in operators if item["name"] == operation)

        result = None

        if operator['type'] == "unary":
            data1 = request.data.get("data1", None)

            if data1 is None:
                return Response(data='Required parameters missing', status=400)
            data1 = parse_data(data1)

            if data1 is None:
                return Response(data='Operands consist of invalid data', status=422)

            result = start_operation(data1=data1, operator=operator)

        elif operator['type'] == "binary":
            data1 = request.data.get("data1", None)
            data2 = request.data.get('data2', None)

            if data1 is None or data2 is None:
                return Response(data='Required parameters missing', status=400)

            data1 = parse_data(data1)
            data2 = parse_data(data2)

            if data1 is None or data2 is None:
                return Response(data='Operands consist of invalid data', status=422)

            result = start_operation(data1=data1, data2=data2, operator=operator)

        if result is None:
            return Response(data='Invalid operation', status=422)

        socket_client = SocketClient().connect()
        socket_client.emit("inform_server", data={"type": "op", "op_id": result.id}, namespace="/ravop")

        response = dict()
        response.update({'op_id': result.id})
        return Response(data=response)


def start_operation(data1=None, data2=None, operator=None):
    from ravop import core
    f = getattr(core, operator["func"])
    if operator['type'] == "binary":
        return f(data1, data2)
    elif operator['type'] == "unary":
        return f(data1)


class Result(APIView):
    def get(self, request, op_id):
        response = dict()
        response.update({'status': None})
        response.update({'result': None})

        try:
            op_object = Op(id=op_id)
        except Exception as e:
            op_object = None
        if op_object is None:
            return Response(data=response, status=404, exception=ObjectDoesNotExist)

        response.update({'status': op_object.status})
        if op_object.status == 'computing' or op_object.status == 'pending':
            return Response(data=response)

        output = op_object.output
        if op_object.output_dtype == "ndarray":
            output = output.tolist()
            response.update({'result': output})

        return Response(data=response)


class ComputeLogisticRegression(APIView):
    def post(self, request):
        try:
            data1 = request.data['data1']
            data2 = request.data['data2']
        except Exception:
            return Response(data='Required parameters missing', status=400)

        data1, type1 = parse_data(data1)
        data2, type2 = parse_data(data2)

        if data1 == 'invalid_data' or data2 == 'invalid_data':
            return Response(data='Operands consist of invalid data', status=422)

        lr = LogisticRegression()
        lr.train(X=data1, y=data2, iter=100)

        socket_client = SocketClient().connect()
        socket_client.emit("update_server", data=None, namespace="/ravop")

        response = dict()
        response.update({ 'id': lr.id })

        return Response(data=response)


class StatusLogisticRegression(APIView):
    def get(self, request, id):
        print("Hey")
        try:
            lr = LogisticRegression(id=id)
        except Exception as e:
            lr = None
        if lr is None:
            return Response(data='Object does not exist', status=404, exception=ObjectDoesNotExist)

        response = lr.get_op_stats()
        response.update(
            {
                'percentage':
                    (response["computed_ops"] + response["computing_ops"] + response["failed_ops"]) /
                    response["total_ops"] * 100
            }
        )
        return Response(data=response)


class PredictLogisticRegression(APIView):
    def post(self, request, id):
        response = dict()
        response.update({'result': None})

        try:
            lr = LogisticRegression(id=id)
        except Exception as e:
            lr = None
        if lr is None:
            return Response(data='Object does not exist', status=404, exception=ObjectDoesNotExist)

        try:
            data1 = request.data['data1']
        except Exception:
            return Response(data='Required parameters missing', status=400)

        data1, type1 = parse_data(data1)
        if data1 == 'invalid_data':
            return Response(data='Operands consist of invalid data', status=422)

        result = lr.predict(data1)
        if type(result) == 'integer':
            response['result'] = result
        else:
            response['result'] = result.tolist()

        return Response(data=response)


class TrainLinearRegression(APIView):
    def post(self, request):
        # Get data type
        data_format = request.data.get("data_format", None)

        if data_format is None:
            return Response(data="Data format is missing", status=400)

        if data_format == "file":

            target_column = request.data.get("target_column", None)

            if target_column is None:
                return Response(data="Target column is missing", status=400)

            # Get the file
            all_files = request.FILES.getlist('file')
            if len(all_files) == 0:
                return Response(data="File is missing", status=400)
            uploaded_file = all_files[0]
            file_name = uploaded_file.name
            file_name = re.sub('[^A-Za-z0-9.]+', '_', file_name)
            file_name = "{}_{}".format(time.time(), file_name)

            os.makedirs(settings.DATASETS_DIR, exist_ok=True)

            file_path = os.path.join(settings.DATASETS_DIR, file_name)
            fout = open(file_path, 'wb+')

            # Iterate through the chunks.
            for chunk in uploaded_file.chunks():
                fout.write(chunk)
            fout.close()

            df = pd.read_csv(file_path)

            y = df[target_column].values
            df.drop(columns=[target_column], inplace=True)
            X = df.values

            print(X, y)
            print(type(X), type(y))

            lr = LinearRegression()
            lr.train(X, y)

            socket_client = SocketClient().connect()
            socket_client.emit("inform_server", data={"type": "graph", "graph_id": lr.id}, namespace="/ravop")

            response = dict()
            response.update({'id': lr.id, "message":"Training started"})
            return Response(data=response)

        elif data_format == "matrices":
            # Get X, get y
            X = request.data.get("X", None)
            y = request.data.get("y", None)

            if X is None or y is None:
                return Response(data="Either X or y is missing", status=400)

            try:
                X = json.loads(X)
                y = json.loads(y)
            except Exception as e:
                return Response(data="Data format of X or y is invalid:{}".format(str(e)), status=400)

            lr = LinearRegression()
            lr.train(X, y)

            socket_client = SocketClient().connect()
            socket_client.emit("inform_server", data={"type": "graph", "graph_id": lr.id}, namespace="/ravop")

            response = dict()
            response.update({'id': lr.id, "message": "Training started"})
            return Response(data=response)

        elif data_format == "dataset":
            dataset = request.data.get("dataset", None)

            if dataset is None:
                return Response(data="Dataset is missing", status=400)

            if dataset == "boston_house_prices":
                X, y = load_boston(return_X_y=True)
            elif dataset == "breast_cancer":
                X, y = load_breast_cancer(return_X_y=True)
            else:
                return Response(data="Invalid dataset", status=400)

            lr = LinearRegression()
            lr.train(X, y)

            socket_client = SocketClient().connect()
            socket_client.emit("inform_server", data={"type": "graph", "graph_id": lr.id}, namespace="/ravop")

            response = dict()
            response.update({'id': lr.id, "message": "Training started"})
            return Response(data=response)


class StatusLinearRegression(APIView):
    def get(self, request, id):
        print("Hey")
        try:
            lr = LinearRegression(id=id)
        except Exception as e:
            lr = None

        if lr is None:
            return Response(data='Invalid id', status=404, exception=ObjectDoesNotExist)

        return Response(data={"progress": lr.progress})


class PredictLinearRegression(APIView):
    def post(self, request, id):
        response = dict()
        response.update({'result': None})

        try:
            lr = LinearRegression(id=id)
        except Exception as e:
            lr = None

        if lr is None:
            return Response(data='Invalid id', status=404, exception=ObjectDoesNotExist)

        data1 = request.data.get('data1', None)

        if data1 is None:
            return Response(data='Required parameters missing', status=400)

        data1 = parse_data(data1)

        if data1 is None:
            return Response(data='Operands consist of invalid data', status=422)

        result = lr.predict(data1)

        if type(result) == 'integer':
            response['result'] = result
        else:
            response['result'] = result.tolist()

        return Response(data=response)
