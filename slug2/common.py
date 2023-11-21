from enum import Enum, auto
from typing import Any

PythonNumber = complex | float | int

UINT8_MAX = 256
FRAMES_MAX = 64
STACK_MAX = FRAMES_MAX * UINT8_MAX


class LocalIndex(int):
    pass


class StackIndex(int):
    pass


class UpvalueIndex(int):
    pass


class ConstantIndex(int):
    pass


class NumArgs(int):
    pass


class JumpDistance(int):
    pass


class ParseError(Exception):
    pass


class CompilerError(Exception):
    pass


def check_number(a: Any) -> bool:
    return isinstance(a, (int, float, complex))


class Op(Enum):
    CONSTANT = auto()

    ADD = auto()
    SUBTRACT = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    INT_DIVIDE = auto()
    EXPONENT = auto()

    TRUE = auto()
    FALSE = auto()

    DEFINE_GLOBAL = auto()
    SET_GLOBAL = auto()
    GET_GLOBAL = auto()

    GET_LOCAL = auto()
    SET_LOCAL = auto()

    CLOSE_UPVALUE = auto()
    GET_UPVALUE = auto()
    SET_UPVALUE = auto()

    # GET_PROPERTY = auto()
    # SET_PROPERTY = auto()
    # GET_SUPER = auto()

    VALUE_EQUAL = auto()
    NOT_VALUE_EQUAL = auto()
    # REFERENCE_EQUAL = auto()
    GREATER = auto()
    GREATER_EQUAL = auto()
    LESS = auto()
    LESS_EQUAL = auto()
    NOT = auto()
    NEGATE = auto()

    JUMP = auto()
    JUMP_IF_FALSE = auto()
    JUMP_FAKE = auto()

    # LOOP = auto()
    CALL = auto()
    # INVOKE = auto()
    # SUPER_INVOKE = auto()
    # CLOSURE = auto()

    NONE = auto()
    POP = auto()
    RETURN = auto()
    ASSERT = auto()
    PRINT = auto()
    CLOSURE = ()
    # CLASS = auto()
    # INHERIT = auto()
    # METHOD = auto()

    def evaluate_binary(self, left: PythonNumber, right: PythonNumber) -> PythonNumber | bool:
        print(left, right)

        if not check_number(left) or not check_number(right):
            raise RuntimeError("invalid number")

        # TODO should we be able to add an int and float? (implicit conversion?)
        #      maybe just for numbers?
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
                return left**right
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


Code = Op | ConstantIndex | LocalIndex | UpvalueIndex | JumpDistance | NumArgs | bool


class FuncType(Enum):
    FUNCTION = auto()
    INITIALIZER = auto()
    METHOD = auto()
    SCRIPT = auto()
