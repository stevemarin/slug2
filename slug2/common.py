from enum import Enum, auto
from typing import Any

PythonNumber = complex | float | int


class LocalIndex(int):
    pass


class ConstantIndex(int):
    pass


class JumpDistance(int):
    pass


class ParseError(Exception):
    pass


class CompilerError(Exception):
    pass


def check_number(a: Any) -> bool:
    return isinstance(a, (int, float, complex))


class FuncType(Enum):
    FUNCTION = auto()
    INITIALIZER = auto()
    METHOD = auto()
    SCRIPT = auto()
