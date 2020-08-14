import app.models as models
import json
import pickle


def get_result(op_object):
    data_id = json.loads(op_object.outputs)[0]

    data = models.Data.objects.get(pk=data_id)
    file_path = data.file_path

    with open(file_path, "rb") as f:
        a = pickle.load(f)

        if data.type == "ndarray":
            return a.tolist()
        else:
            return a
