from common import db
from ravop.core import Op


def get_ops_by_name(op_name, graph_id=None):
    ops = db.get_ops_by_name(op_name=op_name, graph_id=graph_id)

    return [Op(id=op.id) for op in ops]
