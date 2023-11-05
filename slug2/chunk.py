from enum import Enum, auto
from typing import Any

from slug2.common import ConstantIndex, PythonNumber


def check_number(a: Any) -> bool:
    return isinstance(a, (int, float, complex))


class Op(Enum):
    CONSTANT = auto()

    ADD = auto()
    SUBTRACT = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    EXPONENT = auto()

    TRUE = auto()
    FALSE = auto()

    # GET_LOCAL = auto()
    # SET_LOCAL = auto()
    # GET_GLOBAL = auto()
    # DEFINE_GLOBAL = auto()
    # SET_GLOBAL = auto()
    # GET_UPVALUE = auto()
    # SET_UPVALUE = auto()
    # GET_PROPERTY = auto()
    # SET_PROPERTY = auto()
    # GET_SUPER = auto()

    VALUE_EQUAL = auto()
    NOT_VALUE_EQUAL = auto()
    REFERENCE_EQUAL = auto()
    GREATER = auto()
    GREATER_EQUAL = auto()
    LESS = auto()
    LESS_EQUAL = auto()
    NOT = auto()
    NEGATE = auto()

    # PRINT = auto()
    # JUMP = auto()
    # JUMP_IF_FALSE = auto()
    # LOOP = auto()
    # CALL = auto()
    # INVOKE = auto()
    # SUPER_INVOKE = auto()
    # CLOSURE = auto()
    # CLOSE_UPVALUE = auto()

    NOOP = auto()
    POP = auto()
    RETURN = auto()
    # CLASS = auto()
    # INHERIT = auto()
    # METHOD = auto()
    
    ASSERT = auto()

    def evaluate_binary(self, left: PythonNumber, right: PythonNumber) -> PythonNumber | bool:
        if not check_number(left) or not check_number(right):
            raise RuntimeError("invalid number")

        # if type(left) != type(right):
        #     raise RuntimeError("types don't match")

        match self:
            case Op.ADD:
                return left + right
            case Op.SUBTRACT:
                return left - right
            case Op.MULTIPLY:
                return left * right
            case Op.DIVIDE:
                return left / right
            case Op.EXPONENT:
                return left ** right
            case Op.LESS:
                if isinstance(left, complex) or isinstance(right, complex):
                    raise RuntimeError("cannot use < to compare complex numbers")
                return left < right
            case Op.LESS_EQUAL:
                if isinstance(left, complex) or isinstance(right, complex):
                    raise RuntimeError("cannot use <= to compare complex numbers")
                return left <= right
            case Op.GREATER:
                if isinstance(left, complex) or isinstance(right, complex):
                    raise RuntimeError("cannot use > to compare complex numbers")
                return left > right
            case Op.GREATER_EQUAL:
                if isinstance(left, complex) or isinstance(right, complex):
                    raise RuntimeError("cannot use >= to compare complex numbers")
                return left >= right
            case Op.VALUE_EQUAL:
                return left == right
            case Op.NOT_VALUE_EQUAL:
                return left != right
            case _:
                raise RuntimeError("invalid binary op")


class Chunk:
    __slots__ = ("code", "lines", "constants")

    def __init__(self) -> None:
        self.code: list[Op | ConstantIndex] = []
        self.lines: list[int] = []
        self.constants: list[Any] = []

    def write(self, op: Op | ConstantIndex, line: int) -> None:
        self.code.append(op)
        self.lines.append(line)

    def add_constant(self, value: Any) -> int:
        self.constants.append(value)
        return len(self.constants) - 1
