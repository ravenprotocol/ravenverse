from database.db_manager import Graph


def create_graph(db):
    # Create a graph
    graph = Graph()
    graph = db.add(graph)
    return graph
