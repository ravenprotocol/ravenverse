import ast
import sys
from argparse import ArgumentParser
import os
import pandas as pd

from ravpy.federated import compute
from ravpy.globals import g
from ravpy.utils import get_graphs, print_graphs, get_graph, get_subgraph_ops, apply_rules

if __name__ == '__main__':
    argparser = ArgumentParser()

    argparser.add_argument("-a", "--action", type=str, help="Enter action", default=None)
    argparser.add_argument("-c", "--cid", type=str, help="Enter client id", default=None)
    argparser.add_argument("-g", "--graph_id", type=str, help="Id of the graph", default=None)
    argparser.add_argument("-d", "--data", type=str, help="Data to use", default=None)
    argparser.add_argument("-f", "--file_path", type=str, help="File path containing samples to use", default=None)

    if len(sys.argv) == 1:
        argparser.print_help(sys.stderr)
        sys.exit(1)

    args = argparser.parse_args()

    if args.action is None:
        raise Exception("Enter action")

    if args.action == "list":
        graphs = get_graphs()
        print_graphs(graphs)

    elif args.action == "participate":
        if args.cid is None:
            raise Exception("Client id is required")

        if args.graph_id is None:
            # raise Exception("Enter id of the graph to join")
            approach = "distributed"
            print("Participating in Distributed Computing")
        else:
            approach = "federated"
            print("Participating in Federated Analytics")

        # connect
        g.cid = args.cid
        client = g.client
        from ravpy.distributed.benchmarking import benchmark

        if client is None:
            g.client.disconnect()
            raise Exception("Unable to connect to ravsock. Make sure you are using the right hostname and port")

        # Connect
        if approach =="distributed":
            download_path = "./ravpy/distributed/downloads/"
            temp_files_path = "./ravpy/distributed/temp_files/"
            if not os.path.exists(download_path):
                os.makedirs(download_path)
            if not os.path.exists(temp_files_path):
                os.makedirs(temp_files_path)

            bm_results = benchmark()
            print("Benchmark Results: ", bm_results)


        elif approach == "federated":
            graph = get_graph(graph_id=args.graph_id)
            if graph is None:
                g.client.disconnect()
                raise Exception("Invalid graph id")

            subgraph_ops = get_subgraph_ops(graph["id"], cid=args.cid)
            graph_rules = ast.literal_eval(graph['rules'])

            if args.data is None and args.file_path is None:
                g.client.disconnect()
                raise Exception("Provide values or file path to use")

            if args.data is not None:
                pass
                # data_columns = args.data
                # data_columns = ast.literal_eval(args.data)
                # data_silo = apply_rules(data_columns, rules=graph_rules)
                # print(data_silo)
                # if data_silo is not None:
                #     compute(data_silo, graph, subgraph_ops, final_column_names)
                # else:
                #     print("You can't participate as your data is it in the wrong format")

            elif args.file_path is not None:
                print("File path:", args.file_path)
                dataframe = pd.read_csv(args.file_path)
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