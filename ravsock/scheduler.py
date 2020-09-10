
"""
1. Scheduler
2. Distributor

1. One client, one graph - Sequential
2. One client, multiple graphs - 2 Graphs - Sequential and op switching based on complexity and priority
3. Multiple clients, one graph(Multiple ops or single op) - Parallel distribution
4. Multiple clients, multiple graphs - Parallel and op switching based on complexity and priority

(1-10)
Graph1 - User1 - 5 - 6
Graph2 - User2 - 7 - 10

Graph Complexity
1. Number of operations
2. The total complexity of all ops
"""
from common import db


class Scheduler():
    def __init__(self):
        pass

        client = []
        ops = []


