import app.models as models
from app.utils import evaluate

from django.core.exceptions import ObjectDoesNotExist
from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import render

from ravop.ml.core import Core


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
        response = dict()
        response.update({'op_id', output_id})
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
            return Response(data=response, status=204)

        result = evaluate.get_result(op_object)
        response.update({'result': result})

        return Response(data=response)
