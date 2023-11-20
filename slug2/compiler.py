from typing import TYPE_CHECKING, Any

from slug2.chunk import Code, JumpDistance, Op
from slug2.common import CompilerError, ConstantIndex, FuncType, LocalIndex
from slug2.object import ObjFunction
from slug2.token import Token, TokenType

__max_locals__ = 256

if TYPE_CHECKING:
    from slug2.vm import VM


class Local:
    __slots__ = ("name", "depth", "captured")

    def __init__(self, name: Token):
        self.name: Token = name
        self.depth: int = -1
        self.captured: bool = False

    def capture(self):
        self.captured = True

    def __repr__(self) -> str:
        return f"<Local :name {self.name.literal} :depth {self.depth} :value {self.name.value}>"


class Compiler:
    __slots__ = ("vm", "enclosing", "functype", "function", "scope_depth", "locals")

    def __init__(self, vm: "VM", functype: FuncType, root: bool = False):
        self.vm: "VM" = vm

        try:
            self.enclosing: "Compiler | None" = vm.compiler
        except AttributeError:
            self.enclosing = None

        self.functype = functype
        self.function: ObjFunction = ObjFunction()
        self.scope_depth: int = 0
        self.locals: list[Local] = []

        if functype != FuncType.SCRIPT:
            name = self.vm.parser.peek(-1)
        else:
            name = Token(TokenType.IDENTIFIER, "SCRIPT", None, 0, 0)

        self.locals.append(Local(name))

    def compile(self) -> ObjFunction | None:
        self.vm.parser.current_index += 1
        while not self.vm.parser.match(TokenType.EOF):
            self.vm.parser.declaration()

        return None if self.vm.parser.had_error else self.end()

    def end(self) -> ObjFunction | None:
        self.emit_return()
        function = self.function

        self.vm.compiler = self.vm.compiler.enclosing if self.vm.compiler else None

        return function

    def begin_scope(self):
        self.scope_depth += 1

    def end_scope(self, line: int):
        self.scope_depth -= 1

        local_idx = len(self.locals) - 1
        while local_idx >= 0 and self.locals[local_idx].depth > self.scope_depth:
            if self.locals[local_idx].captured:
                self.emit_byte(Op.CLOSE_UPVALUE)
            else:
                self.emit_byte(Op.POP)

            _ = self.locals.pop()
            local_idx -= 1

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

        self.locals[len(self.locals) - 1].depth = self.scope_depth

    def define_variable(self, name: Token, global_index: ConstantIndex) -> None:
        if __debug__:
            print("define_variable", self.locals, self.scope_depth)

        if self.scope_depth > 0:
            self.mark_initialized()
            return

        self.emit_bytes(Op.DEFINE_GLOBAL, global_index)

    def resolve_local(self, name: Token) -> None | LocalIndex:
        for idx, local in enumerate(reversed(self.locals)):
            if name.literal == local.name.literal:
                if local.depth == -1:
                    raise CompilerError("cannot use local variable in own initializer")
                # TODO reconsider this index
                return LocalIndex(len(self.locals) - 1 - idx)
        return None

    def emit_byte(self, op: Code) -> None:
        self.function.chunk.write(op, self.vm.parser.peek(-1).line)

    def emit_bytes(self, op1: Op, op2: Code) -> None:
        self.emit_byte(op1)
        self.emit_byte(op2)

    def make_constant(self, value: Any) -> ConstantIndex:
        chunk = self.vm.compiler.function.chunk if self.vm.compiler else None
        if chunk is None:
            raise CompilerError("chunk is None")
        return chunk.add_constant(value)

    def identifier_constant(self, name: Token) -> ConstantIndex:
        return self.make_constant(name.literal)

    def emit_constant(self, value: Any) -> None:
        constant_index = self.make_constant(value)
        self.emit_bytes(Op.CONSTANT, constant_index)

    def emit_return(self) -> None:
        self.emit_bytes(Op.TRUE, Op.RETURN)

    def emit_jump(self, op: Op) -> JumpDistance:
        self.emit_byte(op)
        self.emit_byte(Op.JUMP_FAKE)
        return JumpDistance(len(self.function.chunk.code) - 1)

    def patch_jump(self, offset: JumpDistance):
        self.function.chunk.code[offset] = JumpDistance(len(self.function.chunk.code) - offset - 1)
