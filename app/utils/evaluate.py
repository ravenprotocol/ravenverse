import app.models as models
import numpy as np
import json


def get_result(op_object):
    data_id = json.loads(op_object.outputs)[0]

    data = models.Data.objects.get(pk=data_id)
    file_path = data.file_path

    with open(file_path, "rb") as f:
        a = json.load(f)

        if isinstance(a, np.ndarray):
            a = a.tolist()
        return a
