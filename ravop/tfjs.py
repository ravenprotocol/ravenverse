from ravop.core import __create_math_op2


def make_method(name):
    def _method(op, **kwargs):
        print("method {0}".format(name))
        return __create_math_op2(op, operator=name, **kwargs)
    return _method


sigmoid = make_method("sigmoid")
sin = make_method("sin")
sinh = make_method("sinh")

