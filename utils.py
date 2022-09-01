import os

import requests
from terminaltables import AsciiTable


def get_my_graphs():
    # Get graphs
    headers = {"token": os.environ.get("TOKEN")}
    r = requests.get(url="{}/graph/all/".format(os.environ.get("RAVENVERSE_URL")), headers=headers)

    if r.status_code != 200:
        print("Error:{}".format(r.text))
        return None

    graphs = r.json()
    table_data = [["Id", "Name", "Approach", "Algorithm", "Rules", "Status", "Cost"]]
    for graph in graphs:
        table_data.append([graph['id'], graph['name'], graph['approach'], graph['algorithm'], graph['rules'],
                           graph['status'], str(graph['cost'])+" Tokens"])
    print("Graphs:\n", AsciiTable(table_data).table)

    return graphs
