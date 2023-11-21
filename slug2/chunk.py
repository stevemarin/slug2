from typing import Any

from slug2.common import Code, ConstantIndex, JumpDistance


class Chunk:
    __slots__ = ("code", "lines", "constants")

    def __init__(self) -> None:
        self.code: list[Code] = []
        self.lines: list[int] = []
        self.constants: list[Any] = []

    def write(self, op: Code, line: int) -> None:
        self.code.append(op)
        self.lines.append(line)

    def add_constant(self, value: Any) -> ConstantIndex:
        self.constants.append(value)
        return ConstantIndex(len(self.constants) - 1)

    def __repr__(self) -> str:
        repr = "\nCHUNK:\n"
        for code in self.code:
            if isinstance(code, ConstantIndex):
                repr += f"  {code} -> {self.constants[code]}\n"
            elif isinstance(code, JumpDistance):
                repr += f"  jump -> {code}\n"
            else:
                repr += f"  {code}\n"
        return repr
