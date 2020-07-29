import glob
import os


def clean():
    for file_path in glob.glob("files/*"):
        if os.path.exists(file_path):
            os.remove(file_path)

    if os.path.exists("ravop_log.log"):
        os.remove("ravop_log.log")

    if os.path.exists("ravsock_log.log"):
        os.remove("ravsock_log.log")

    if not os.path.exists("files"):
        os.makedirs("files")


if __name__ == '__main__':
    clean()

