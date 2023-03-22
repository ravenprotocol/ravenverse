import os

from dotenv import load_dotenv

load_dotenv()

from ravpy.initialize import initialize
from ravpy.distributed.participate import participate
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_id", help="Set graph id to participate", type=int)

    args = parser.parse_args()
    client = initialize(os.environ.get("TOKEN"))
    participate(graph_id=args.graph_id)
