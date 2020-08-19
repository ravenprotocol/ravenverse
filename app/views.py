import app.models as models
import app.constants as constants
from app.utils import evaluate

from django.core.exceptions import ObjectDoesNotExist
from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import render

from ravop.ml.core import Core
import json
import numpy as np


def home(request):
    return render(request, "home.html")


def get_operator_name(operator_number):
    operators = [e.value for e in constants.Operators]
    return operators[operator_number - 1]


def parse_data(data):
    if not data:
        return data, None
    try:
        return int(data), 'integer'
    except ValueError:
        try:
            return float(data), 'double'
        except ValueError:
            try:
                data = json.loads(data)
                data = np.array(data)
                return data, 'ndarray'
            except ValueError:
                return 'invalid_data', ''


class Compute(APIView):
    def post(self, request):
        try:
            data1 = request.data['data1']
            data2 = request.data['data2']
            operator_number = request.data['operation']
        except Exception:
            return Response(data='Required parameters missing', status=400)

        data1, type1 = parse_data(data1)
        data2, type2 = parse_data(data2)

        if data1 == 'invalid_data' or data2 == 'invalid_data':
            return Response(data='Operands consist of invalid data', status=422)

        if operator_number <= 5:
            data_list = [{"name": "x", "data": data1, "type": type1},
                         {"name": "y", "data": data2, "type": type2}]
            op_type = 'binary'

        else:
            data_list = [{"name": "x", "data": data1, "type": type1}]
            op_type = 'unary'

        core = Core(graph_id=1)
        output_id = core.compute(data_list=data_list, operator=get_operator_name(operator_number), op_type=op_type)

        response = dict()
        response.update({'op_id': output_id})
        return Response(data=response)


class Result(APIView):
    def get(self, request, op_id):
        response = dict()
        response.update({'status': None})
        response.update({'result': None})

        try:
            op_object = models.Op.objects.get(pk=op_id)
        except ObjectDoesNotExist as e:
            op_object = None

        if op_object is None:
            return Response(data=response, status=404, exception=ObjectDoesNotExist)

        response.update({'status': op_object.status})
        if op_object.status == 'computing' or op_object.status == 'pending':
            return Response(data=response)

        result = evaluate.get_result(op_object)
        response.update({'result': result})

        return Response(data=response)
