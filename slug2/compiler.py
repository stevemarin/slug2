from typing import TYPE_CHECKING, Any

from slug2.common import (
    UINT8_MAX,
    Code,
    CompilerError,
    ConstantIndex,
    FuncType,
    JumpDistance,
    LocalIndex,
    Op,
    UpvalueIndex,
)
from slug2.object import ObjFunction, ObjType
from slug2.token import Token, TokenType

if TYPE_CHECKING:
    from slug2.vm import VM


class Upvalue:
    def __init__(self, index: LocalIndex | UpvalueIndex, is_local: bool):
        self.index = index
        self.is_local = is_local

    def __repr__(self) -> str:
        return f"<Upvalue :is_local {self.is_local} :value {self.index}>"


class Local:
    __slots__ = ("name", "depth", "captured")

    def __init__(self, name: Token):
        self.name: Token = name
        self.depth: int = -1
        self.captured: bool = False

    def __repr__(self) -> str:
        return f"<Local :name {self.name.literal} :depth {self.depth} :value {self.name.value}>"


class Compiler:
    __slots__ = (
        "vm",
        "name",
        "enclosing",
        "functype",
        "function",
        "scope_depth",
        "num_locals",
        "locals",
        "num_upvalues",
        "upvalues",
    )

    def __init__(self, vm: "VM", funcname: str, functype: FuncType):
        self.vm: "VM" = vm
        self.name = funcname
        self.functype = functype
        self.scope_depth: int = 0
        self.num_locals: int = 0
        self.locals: list["Local | None"] = [None] * UINT8_MAX
        self.num_upvalues: int = 0
        self.upvalues: list["Upvalue | None"] = [None] * UINT8_MAX

        try:
            self.enclosing: "Compiler | None" = vm.compiler
        except AttributeError:
            self.enclosing = None  # root compiler

        vm.compiler = self
        self.function: ObjFunction = ObjFunction(vm, ObjType.FUNCTION, functype)

        if functype != FuncType.SCRIPT:
            self.function.name = self.vm.parser.peek(-1).literal

        if functype == FuncType.FUNCTION:
            name = Token(TokenType.IDENTIFIER, "this", None, 0, 0)
        else:
            name = Token(TokenType.IDENTIFIER, "", None, 0, 0)

        self.locals[self.num_locals] = Local(name)
        self.num_locals += 1

    def compile(self) -> ObjFunction | None:
        self.vm.parser.current_index += 1
        while not self.vm.parser.match(TokenType.EOF):
            self.vm.parser.declaration()

        return None if self.vm.parser.had_error else self.end()

    def end(self) -> ObjFunction:
        self.emit_return()
        function = self.function

        if self.enclosing is not None:
            self.vm.compiler = self.enclosing
        else:
            # TODO what to do when enclosing is None, just for root?
            self.vm.compiler = Compiler(self.vm, "", FuncType.NONE)

        return function

    def begin_scope(self):
        self.scope_depth += 1

    def end_scope(self):
        self.scope_depth -= 1

        while self.num_locals > 0:
            local = self.locals[self.num_locals - 1]
            if local is None:
                raise RuntimeError("ending scope - local is None")
            elif local.depth <= self.scope_depth:
                break
            elif local.captured:
                self.emit_byte(Op.CLOSE_UPVALUE)
            else:
                self.emit_byte(Op.POP)

            self.num_locals -= 1

    def add_upvalue(self, index: LocalIndex | UpvalueIndex, is_local: bool) -> UpvalueIndex:
        # TODO remove bool and just use type of index
        print("ADDING UPVALUE:", self.function.num_upvalues, index, is_local)

        for idx, upvalue in enumerate(self.upvalues):
            upvalue = self.upvalues[idx]
            if upvalue is None:
                break
            elif upvalue.index == index and upvalue.is_local == is_local:
                return UpvalueIndex(idx)

        if self.function.num_upvalues == UINT8_MAX:
            raise CompilerError("too many closure variables in function")

        self.upvalues[self.function.num_upvalues] = Upvalue(index, is_local)
        self.function.num_upvalues += 1
        print("NUM_UPVALUES:", self.function.num_upvalues, self.upvalues[self.function.num_upvalues-1])
        return UpvalueIndex(self.function.num_upvalues - 1)

    def resolve_upvalue(self, name: Token) -> UpvalueIndex | None:
        if self.enclosing is None:
            return None

        local_index = self.enclosing.resolve_local(name)
        if local_index is not None:
            local = self.enclosing.locals[local_index]
            if local is None:
                raise RuntimeError("local is none")
            local.captured = True
            return self.add_upvalue(local_index, True)

        upvalue = self.enclosing.resolve_upvalue(name)
        if upvalue is not None:
            return self.add_upvalue(upvalue, False)

        return None

    def add_local(self, name: Token) -> None:
        if self.num_locals == UINT8_MAX:
            raise CompilerError("too many locals")

        self.locals[self.num_locals] = Local(name)
        self.num_locals += 1

    def declare_variable(self, name: Token):
        if self.scope_depth == 0:
            return

        for idx in range(self.num_locals - 1, -1, -1):
            local = self.locals[idx]
            if local is None:
                continue
            elif local.depth != -1 and local.depth < self.scope_depth:
                break

            if name.literal == local.name.literal:
                raise CompilerError("a variable exists with same name in this scope")

        self.add_local(name)

    def mark_initialized(self) -> None:
        if self.scope_depth == 0:
            return

        local = self.locals[self.num_locals - 1]
        if local is None:
            raise RuntimeError("local is none")
        else:
            local.depth = self.scope_depth

    def define_variable(self, global_index: ConstantIndex) -> None:
        if __debug__:
            assert all([_ is None for _ in self.locals[self.num_locals:]])
            print("define_variable", self.locals[self.num_locals], self.scope_depth)

        if self.scope_depth > 0:
            self.mark_initialized()
            return

        self.emit_bytes(Op.DEFINE_GLOBAL, global_index)

    def resolve_local(self, name: Token) -> None | LocalIndex:
        for idx in range(self.num_locals - 1, -1, -1):
            print("\t -> idx", idx)
            local = self.locals[idx]
            if local is None:
                raise RuntimeError("cannot resolve local - local is None")
            elif name.literal == local.name.literal:
                if local.depth == -1:
                    raise CompilerError("cannot use local variable in own initializer")
                return LocalIndex(idx)
        return None

    def emit_byte(self, op: Code) -> None:
        self.function.chunk.write(op, self.vm.parser.peek(-1).line)

    def emit_bytes(self, op1: Op, op2: Code) -> None:
        self.emit_byte(op1)
        self.emit_byte(op2)

    def emit_upvalue(self, op1: bool, op2: LocalIndex | UpvalueIndex) -> None:
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
        if self.function.functype == FuncType.INITIALIZER:
            self.emit_bytes(Op.GET_LOCAL, ConstantIndex(0))
        else:
            self.emit_byte(Op.NONE)

        self.emit_byte(Op.RETURN)

    def emit_jump(self, op: Op) -> JumpDistance:
        self.emit_byte(op)
        self.emit_byte(Op.JUMP_FAKE)
        return JumpDistance(len(self.function.chunk.code) - 1)

    def patch_jump(self, offset: JumpDistance):
        self.function.chunk.code[offset] = JumpDistance(len(self.function.chunk.code) - offset - 1)
