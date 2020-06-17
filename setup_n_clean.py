import glob
import os
from database.db_manager import DBManager


def clean():
    for file_path in glob.glob("files/*"):
        if os.path.exists(file_path):
            os.remove(file_path)

    # Remove db
    if os.path.exists("raven_db.db"):
        os.remove("raven_db.db")

    if os.path.exists("logs.out"):
        os.remove("logs.out")

    DBManager().create_tables()


if __name__ == '__main__':
    clean()

