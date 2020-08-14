import app.models as models
from app.utils import evaluate

from django.contrib.auth.decorators import login_required
from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import render

from ravop.ml.core import Core
from ravop.socket_client import SocketClient


def home(request):
    return render(request, "home.html")


class Compute(APIView):
    def post(self, request):
        try:
            data1 = request.data['data1']
            data2 = request.data['data2']
            type1 = request.data['type1']
            type2 = request.data['type2']

        except Exception:
            return Response(data='Required parameters missing', status=400)

        core = Core(graph_id=1)
        data_list = [{"name": "x", "data": data1, "type": type1},
                     {"name": "y", "data": data2, "type": type2}]

        output_id = core.compute(data_list=data_list, operator='addition', op_type='binary')
        return Response('output_id: {}'.format(output_id))


class Result(APIView):
    def get(self, request, op_id):
        op_object = models.Op.objects.get(pk=op_id)
        if op_object.status == 'Computing' or op_object.status == 'Pending':
            return Response(op_object.status)

        op_base = Core(graph_id=1)
        result = evaluate.get_result(op_object)
        return Response('Result: {}'.format(result))
