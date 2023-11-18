from typing import Any

PythonNumber = complex | float | int


class LocalIndex(int):
    pass


class ConstantIndex(int):
    pass


class JumpDistance(int):
    pass


def check_number(a: Any) -> bool:
    return isinstance(a, (int, float, complex))
