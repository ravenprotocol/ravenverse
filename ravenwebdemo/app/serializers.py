import app.models as models
from rest_framework import serializers


class GraphSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Graph
        fields = '__all__'


class DataSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Data
        fields = '__all__'


class ClientSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Client
        fields = '__all__'


class OpSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Op
        fields = '__all__'
