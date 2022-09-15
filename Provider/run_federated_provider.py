import os
from dotenv import load_dotenv
load_dotenv()

from ravpy.utils import list_graphs
from ravpy.federated.participate import participate
from ravpy.initialize import initialize

client = initialize(os.environ.get("TOKEN"))

list_graphs(approach="federated")

participate(graph_id=1, file_path="data/data1.csv")