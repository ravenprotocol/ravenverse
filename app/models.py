from django.db import models
from django.utils import timezone

import app.constants as constants


class Graph(models.Model):
    class Meta:
        db_table = 'graph'

    statuses = [("active", "Active"),
                ("inactive", "Inactive")]
    status = models.CharField(max_length=10, default="active", choices=statuses)
    created_at = models.DateTimeField(default=timezone.now)


class Data(models.Model):
    class Meta:
        db_table = 'data'

    # Required validation choice
    data_types = [(entity.name, entity.value) for entity in constants.DataTypes]

    type = models.CharField(max_length=20, choices=data_types)
    file_path = models.CharField(max_length=500, null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)


class Client(models.Model):
    class Meta:
        db_table = 'client'

    # Required validation choices
    client_types = [(entity.name, entity.value) for entity in constants.ClientTypes]
    status_types = [(entity.name, entity.value) for entity in constants.ClientStatus]

    client_id = models.CharField(max_length=100)
    client_ip = models.CharField(max_length=20, null=True, blank=True)
    status = models.CharField(max_length=20, choices=status_types, default='disconnected')
    type = models.CharField(max_length=10, choices=client_types, null=True, blank=True)
    connected_at = models.DateTimeField(default=timezone.now)
    disconnected_at = models.DateTimeField(default=timezone.now)
    created_at = models.DateTimeField(default=timezone.now)


class Op(models.Model):
    class Meta:
        db_table = 'op'

    # Required validation choices
    node_types = [(entity.name, entity.value) for entity in constants.NodeTypes]
    op_types = [(entity.name, entity.value) for entity in constants.OpTypes]
    operators = [(entity.name, entity.value) for entity in constants.Operators]
    status_types = [(entity.name, entity.value) for entity in constants.OpStatus]

    name = models.CharField(max_length=20, null=True, blank=True)
    graph = models.ForeignKey(Graph, null=True, blank=True, on_delete=models.SET_NULL, related_name='graphs')
    node_type = models.CharField(max_length=100, choices=node_types)

    # Inputs: List of op IDs
    # Output: List of pickle files
    inputs = models.TextField(null=True, blank=True)
    outputs = models.CharField(max_length=300, null=True, blank=True)

    op_type = models.CharField(max_length=100, choices=op_types)
    operator = models.CharField(max_length=100, choices=operators)
    status = models.CharField(max_length=20, choices=status_types, default='pending')

    # List of clients
    client_id = models.TextField(null=True, blank=True)

    created_at = models.DateTimeField(default=timezone.now)
