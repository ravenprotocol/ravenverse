import os
import json
import numpy as np

from .constants import DATA_FILES_PATH


def save_data_to_file(data_id, data):
    """
    Method to save data in a pickle file
    """
    file_path = os.path.join(DATA_FILES_PATH, "data_{}.json".format(data_id))

    if os.path.exists(file_path):
        os.remove(file_path)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        if isinstance(data, np.ndarray):
            data = data.tolist()
        json.dump(data, f)

    return file_path


def delete_data_file(data):
    file_path = os.path.join(DATA_FILES_PATH, "data_{}.json".format(data.id))
    if os.path.exists(file_path):
        os.remove(file_path)
