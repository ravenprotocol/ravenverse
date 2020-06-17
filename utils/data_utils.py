import os
import pickle

import numpy as np

import settings


class DataManager(object):
    def __init__(self, db):
        self.db = db

    def create_data(self, data, data_type):
        # print("Creating data:", data)

        if isinstance(data, (np.ndarray, np.generic)):
            if data.ndim == 1:
                data = data[..., np.newaxis]

        # Create data
        # d = Data()
        # d.type = data_type
        # d = db.add(d)

        d = self.db.create_data(type=data_type)

        # Save file
        file_path = self.save_data_to_file(d.id, data)

        # Update file path
        self.db.update(d, file_path=file_path)

        return d

    def save_data_to_file(self, data_id, data):
        """
        Method to save data in a pickle file
        """
        file_path = os.path.join(settings.DATA_FILES_PATH, "data_{}.pkl".format(data_id))
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                pickle.dump(data, f)

            return file_path
        else:
            return file_path

    def delete_data(self, data):
        file_path = os.path.join(settings.DATA_FILES_PATH, "data_{}.pkl".format(data.id))
        if os.path.exists(file_path):
            os.remove(file_path)

        self.db.delete(data)
