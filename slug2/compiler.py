from enum import Enum, auto

from slug2.object import ObjFunction
from slug2.parser import Parser, emit_return
from slug2.token import Token, TokenType


class Local:
    __slots__ = ("name", "depth", "captured")

    def __init__(self, name: Token):
        self.name: Token = name
        self.depth: int = 0
        self.captured: bool = False

    def capture(self):
        self.captured = True


class FuncType(Enum):
    FUNCTION = auto()
    INITIALIZER = auto()
    METHOD = auto()
    SCRIPT = auto()


class Compiler:
    __slots__ = ("parser", "enclosing", "functype", "function", "scope_depth", "locals")

    def __init__(self, source: str, enclosing: "Compiler | None", functype: FuncType):
        self.parser = Parser(source)
        self.enclosing = enclosing
        self.functype = functype
        self.function = ObjFunction()
        self.scope_depth: int = 0
        self.locals: list[Local] = []

        if functype != FuncType.SCRIPT:
            name = self.parser.peek(-1)
        else:
            name = Token(TokenType.IDENTIFIER, "SCRIPT", None, 0, 0)

        self.locals.append(Local(name))

    def compile(self) -> ObjFunction | None:
        self.parser.current_index += 1
        while not self.parser.match(TokenType.EOF):
            self.parser.declaration()

        function: ObjFunction = self.end(self.parser.peek(-1).line)

        return None if self.parser.had_error else function

    def end(self, line: int) -> ObjFunction:
        emit_return(line)
        return self.function
