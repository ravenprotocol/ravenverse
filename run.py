import os

import argparse
from dotenv import load_dotenv
load_dotenv()

from utils import get_my_graphs
import ravop as R

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--action', type=str, default="list",
                        help='List my graphs/models')

    args = parser.parse_args()

    R.initialize(ravenverse_token=os.environ.get("TOKEN"))

    if args.action == "list":
        get_my_graphs()
