import os

from dotenv import load_dotenv

load_dotenv()

from ravpy.distributed.participate import participate
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_id", help="Set graph id to participate", type=int)

    args = parser.parse_args()
    participate(token=os.environ.get("TOKEN"), graph_id=args.graph_id)
