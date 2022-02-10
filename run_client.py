import ast
from argparse import ArgumentParser

import pandas as pd

from ravpy.federated import compute
from ravpy.globals import g
from ravpy.utils import get_graphs, print_graphs, get_graph, get_subgraph_ops, apply_rules

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--action", type=str, help="Enter action", default=None)
    argparser.add_argument("--cid", type=str, help="Enter client id", default=None)
    argparser.add_argument("--federated_id", type=str, help="Id of the federated graph", default=None)

    args = argparser.parse_args()

    if args.action is None:
        raise Exception("Enter action")

    if args.action == "list":
        graphs = get_graphs()
        print_graphs(graphs)

    elif args.action == "participate":
        if args.cid is None:
            raise Exception("Client id is required")

        if args.federated_id is None:
            raise Exception("Enter id of the federated analytics graph to join")

        print("Let's participate")

        # connect
        g.cid = args.cid
        client = g.client

        if client is None:
            print("Unable to connect to ravsock. Make sure you are using the right hostname and port")
            exit()

        # Connect
        graph = get_graph(graph_id=args.federated_id)
        if graph is None:
            raise Exception("Invalid graph id")

        subgraph_ops = get_subgraph_ops(graph["id"], cid=args.cid)
        graph_rules = ast.literal_eval(graph['rules'])

        # user_choice = input("How do you want to input your data samples?(0: file, 1: other): ")
        #
        # while user_choice not in ["0", "1"]:
        #     user_choice = input("How do you want to input your data samples?(0: file, 1: other): ")
        #
        # if user_choice == "0":
        file_path = input("Enter data file path: ")

        dataframe = pd.read_csv(file_path)
        column_names = []
        for col in dataframe.columns:
            column_names.append(col)
        # column_names.sort()

        final_column_names = []
        for key, value in graph_rules['rules'].items():
            if key in column_names:
                final_column_names.append(key)
            else:
                raise Exception('Incorrect Rules Format.')
        final_column_names.sort()

        data_columns = []
        for column_name in final_column_names:
            column = dataframe[column_name].tolist()
            data_columns.append(column)

        data_silo = apply_rules(data_columns, rules=graph_rules, final_column_names=final_column_names)
        if data_silo is not None:
            compute(args.cid, data_silo, graph, subgraph_ops, final_column_names)
        else:
            print("You can't participate as your data is it in the wrong format")

        # elif user_choice == "1":
        #     data_columns = input("Enter data: ")
        #     data_columns = ast.literal_eval(data_columns)
        #     data_silo = apply_rules(data_columns, rules=graph_rules)
        #     print(data_silo)
        #     if data_silo is not None:
        #         compute(data_silo, graph, subgraph_ops, final_column_names)
        #     else:
        #         print("You can't participate as your data is it in the wrong format")
