import os

from dotenv import load_dotenv

load_dotenv()

from ravpy.distributed.participate import participate
from ravpy.initialize import initialize

if __name__ == '__main__':
    client = initialize(os.environ.get("TOKEN"))
    participate()
