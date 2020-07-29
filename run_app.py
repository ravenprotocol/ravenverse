import json

import numpy as np

from common.db_manager import Operators, OpTypes
from ravop.op_base import OpBase
from ravop.socket_client import SocketClient


def get_operator_name(operator_number):
    operators = [e.value for e in Operators]
    return operators[operator_number - 1]


if __name__ == '__main__':
    socket_client = SocketClient().connect()

    i = 0
    while True:
        op_base = OpBase(graph_id=i, socket_client=socket_client)
        i += 1

        # Create a graph here
        print("\n======================")
        print("Select Operator:")
        print("1. Matrix Multiplication")
        print("2. Addition")
        print("3. Subtraction")
        print("4. Element-wise-multiplication")
        print("5. Division")
        print("6. Linear")
        print("7. Negation")
        print("8. Exponential")
        print("9. Transpose")
        print("10. Natural Log")
        print("11. Quit")

        print("\n")

        operator_number = input("Enter operator number here: ")

        while operator_number == "":
            operator_number = input("Enter operator number here: ")

        operator_number = int(operator_number)

        if operator_number > 10 or operator_number < 1:
            print("Enter a valid operator number")
            continue

        if operator_number <= 5:
            operand1 = input("\nEnter the first value(Scalar, Array or Matrix): ")

            try:
                operand1 = json.loads(operand1)
                operand1 = np.array(operand1)
                operand_type = "ndarray"
            except Exception as e:
                print("Error:", e)
                try:
                    operand1 = int(operand1)
                    operand_type = "integer"
                except Exception as e:
                    print("Error:", e)
                    print("Unsupported operand type")
                    continue

            operand2 = input("\nEnter the second value(Scalar, Array or Matrix): ")

            try:
                operand2 = json.loads(operand2)
                operand2 = np.array(operand2)
                operand_type = "ndarray"
            except Exception as e:
                print("Error:", e)
                try:
                    operand2 = int(operand2)
                    operand_type = "integer"
                except Exception as e:
                    print("Error:", e)
                    print("Unsupported operand type")
                    continue

            # Create a operation
            data_list = [{"name": "x", "data": operand1, "type": operand_type},
                         {"name": "y", "data": operand2, "type": operand_type}]

            op_base.compute(data_list=data_list, operator=get_operator_name(operator_number),
                            op_type=OpTypes.BINARY.value)

            op_base.get_result()
        elif operator_number == 11:
            break
        else:
            operand1 = input("\nEnter the value(Scalar, Array or Matrix): ")

            try:
                operand1 = json.loads(operand1)
                operand1 = np.array(operand1)
                operand_type = "ndarray"
            except Exception as e:
                print("Error:", e)
                try:
                    operand1 = int(operand1)
                    operand_type = "integer"
                except Exception as e:
                    print("Error:", e)
                    print("Unsupported operand type")
                    continue

            # Create a operation
            data_list = [{"name": "x", "data": operand1, "type": operand_type}]

            op_base.compute(data_list=data_list, operator=get_operator_name(operator_number),
                            op_type=OpTypes.UNARY.value)

            op_base.get_result()
