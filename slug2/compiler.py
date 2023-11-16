from enum import Enum, auto

from slug2.common import ConstantIndex, LocalIndex
from slug2.object import ObjFunction
from slug2.parser import Parser, emit_bytes, emit_return
from slug2.token import Token, TokenType

__max_locals__ = 256


class CompilerError(Exception):
    pass


class Local:
    __slots__ = ("name", "depth", "captured")

    def __init__(self, name: Token):
        self.name: Token = name
        self.depth: int = -1
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

    def add_local(self, name: Token) -> None:
        if len(self.locals) == __max_locals__:
            raise CompilerError("too many locals")

        self.locals.append(Local(name))

    def declare_variable(self, name: Token):
        if self.scope_depth == 0:
            return

        for local in reversed(self.locals):
            if local.depth != -1 and local.depth < self.scope_depth:
                break

            if name.literal == local.name.literal:
                raise CompilerError("a variable exists with same name in this scope")

        self.add_local(name)

    def mark_initialized(self) -> None:
        if self.scope_depth == 0:
            return

        self.locals[-1].depth = self.scope_depth

    def define_variable(self, name: Token, global_index: ConstantIndex) -> None:
        if self.scope_depth > 0:
            self.mark_initialized()
            return

        from slug2.chunk import Op

        emit_bytes(Op.DEFINE_GLOBAL, global_index, name.line, name.line)

    def resolve_local(self, name: Token) -> None | LocalIndex:
        for idx, local in enumerate(reversed(self.locals)):
            if name.literal == local.name.literal:
                if local.depth == -1:
                    raise CompilerError("canot real local variable in own initializer")
                return LocalIndex(idx)
        return None
