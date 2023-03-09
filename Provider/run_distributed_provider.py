import os

from dotenv import load_dotenv

load_dotenv()

from ravpy.initialize import initialize
from ravpy.distributed.participate import participate

if __name__ == '__main__':
    client = initialize(os.environ.get("TOKEN"))
    participate(graph_id=1)
