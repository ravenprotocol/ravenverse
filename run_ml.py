import time

from sklearn.datasets import load_boston

from ravop.core import Scalar, Tensor
from ravop.ml import LinearRegression
from ravop.socket_client import SocketClient

if __name__ == '__main__':
    # db = DBManager()
    # b = Scalar(3, db=db)
    # c = Scalar(20, db=db)
    # d = b.add(c)
    #
    # status = d.status
    # while status == "pending" or status == "computing":
    #     db2 = DBManager()
    #     status = d.status
    #     if status == "computed":
    #         print(d.output)
    #     db2.session.close()

    # g = Graph(id=1)
    # print(g)
    # db = DBManager.Instance()
    # db.create_tables()

    # d = Data(value=2, dtype="int")
    # print(d, d.value, type(d.value))
    #
    # t = Tensor(value=[[2, 3, 4]])
    # print(t.output, t.dtype, t.shape)

    # s1 = DBManager.Instance()
    # print(s1)

    # socket_client = SocketClient().connect()
    #
    # from ravop import globals as g
    # g.graph_id = None
    #
    # a = Scalar(10)
    # b = Scalar(20)
    # c = a.elemul(b)

    # socket_client.emit("update_server", data=None, namespace="/ravop")
    #
    # while c.status == "pending" or c.status == "computing":
    #     pass
    #
    # print(c.output)

    # from ravop import globals as g
    # from common import db
    #
    # db.deactivate_all_graphs()
    # db.disconnect_all_clients()
    # exit()

    # a = Tensor([[2, 3, 4, 5], [5, 4, 5, 6], [3, 4, 5, 1]])
    # b = Tensor([[1], [2], [3], [8]])
    # c = a.matmul(b)

    # X, y = load_boston(return_X_y=True)
    X, y = [[2, 3, 4], [4, 5, 6]], [2, 3]

    print(time.time())
    lr = LinearRegression()
    cost, weights = lr.train(X=X, y=y, iter=10)

    socket_client = SocketClient().connect()
    socket_client.emit("inform_server", data={"type": "graph"}, namespace="/ravop")

    print("Waiting for cost")
    while cost.status == "pending" or cost.status == "computing":
        pass

    print("Cost:{}".format(cost.output))
    print("Weights:{}".format(weights.output))
    print(time.time())

