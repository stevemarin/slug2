from typing import TYPE_CHECKING, Any

from slug2.common import (
    UINT8_MAX,
    Code,
    CompilerError,
    ConstantIndex,
    EntryExit,
    FuncType,
    JumpDistance,
    LocalIndex,
    Op,
    UpvalueIndex,
    debug_print,
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

    def __init__(self, vm: "VM", functype: FuncType):
        self.vm: "VM" = vm
        self.enclosing = self.vm.compiler

        self.functype = functype

        self.scope_depth: int = 0

        self.num_locals: int = 0
        self.locals: list["Local | None"] = [None] * UINT8_MAX

        self.num_upvalues: int = 0
        self.upvalues: list["Upvalue | None"] = [None] * UINT8_MAX

        if functype != FuncType.SCRIPT:
            funcname = self.vm.parser.peek(-1).literal
        else:
            funcname = "__main__"

        self.function: ObjFunction = ObjFunction(vm, ObjType.FUNCTION, funcname, functype)

        if functype != FuncType.FUNCTION:
            name = Token(TokenType.IDENTIFIER, "this", None, 0, 0)
        else:
            name = Token(TokenType.IDENTIFIER, "", None, 0, 0)

        self.locals[self.num_locals] = Local(name)
        self.num_locals += 1

        self.vm.compiler = self

    def print_upvalue(self, upvalue: Upvalue) -> None:
        if upvalue.is_local:
            debug_print(self.locals[upvalue.index])
        else:
            debug_print(self.vm.stack[self.vm.stack[upvalue.index]])

    @EntryExit("Compiler.compile")
    def compile(self) -> ObjFunction | None:
        self.vm.parser.current_index += 1
        while not self.vm.parser.match(TokenType.EOF):
            if self.vm.parser.match(TokenType.NEWLINE):
                continue
            self.vm.parser.declaration()

        return None if self.vm.parser.had_error else self.end()

    @EntryExit("Compiler.end")
    def end(self) -> ObjFunction:
        print(self.locals)
        self.emit_return()
        current_function = self.function
        self.vm.compiler = self.enclosing
        return current_function

    @EntryExit("Compiler.begin_scope")
    def begin_scope(self):
        self.scope_depth += 1

    @EntryExit("Compiler.end_scope")
    def end_scope(self):
        self.scope_depth -= 1

        assert len([_ for _ in self.locals if _ is not None]) == self.num_locals

        while self.num_locals > 0:
            local = self.locals[self.num_locals - 1]
            assert local is not None

            if not local.depth > self.scope_depth:
                break

            if local.captured:
                self.emit_byte(Op.CLOSE_UPVALUE)
            else:
                self.emit_byte(Op.POP)

            # TODO do we need top set to none?  goes out of scope right after?
            self.locals[self.num_locals - 1] = None
            self.num_locals -= 1

        assert len([x for x in self.locals if x is not None]) == self.num_locals

    @EntryExit("Compiler.add_upvalue")
    def add_upvalue(self, index: LocalIndex | UpvalueIndex, is_local: bool) -> UpvalueIndex:
        # TODO remove bool and just use type of index
        assert len([x for x in self.upvalues if x is not None]) == self.num_upvalues

        debug_print(f"starting upvalues: {self.function.num_upvalues}")
        for uv in self.upvalues[: self.function.num_upvalues]:
            assert uv is not None
            self.print_upvalue(uv)

        for idx in range(self.function.num_upvalues):
            upvalue = self.upvalues[idx]

            assert upvalue is not None

            if upvalue.index == index and upvalue.is_local == is_local:
                return UpvalueIndex(idx)

        if self.function.num_upvalues == UINT8_MAX:
            raise CompilerError("too many closure variables in function")

        self.upvalues[self.num_upvalues] = Upvalue(index, is_local)
        self.num_upvalues += 1

        assert len([x for x in self.upvalues if x is not None]) == self.num_upvalues
        assert len([x for x in self.locals if x is not None]) == self.num_locals

        debug_print(f"ending upvalues: {self.function.num_upvalues}")
        debug_print(" ".join([str(uv) for uv in self.upvalues if uv is not None]))
        for uv in self.upvalues[: self.function.num_upvalues]:
            assert uv is not None
            self.print_upvalue(uv)

        return UpvalueIndex(self.num_upvalues - 1)

    @EntryExit("Compiler.resolve_upvalue")
    def resolve_upvalue(self, name: Token) -> LocalIndex | UpvalueIndex | None:
        if self.enclosing is None:
            debug_print("returning None")
            return None

        local_index = self.enclosing.resolve_local(name)
        if local_index is not None:
            local = self.enclosing.locals[local_index]
            assert local is not None
            local.captured = True
            idx = self.add_upvalue(local_index, True)
            debug_print(f"returning local index {idx}")
            return idx

        upvalue_index = self.enclosing.resolve_upvalue(name)
        if upvalue_index is not None:
            idx = self.add_upvalue(upvalue_index, False)
            print(f"returning upvalue_index {idx}")
            return idx

        print("returning None")
        return None

    @EntryExit("Compiler.add_local")
    def add_local(self, name: Token) -> None:
        if self.num_locals == UINT8_MAX:
            raise CompilerError("too many locals")

        self.locals[self.num_locals] = Local(name)
        self.num_locals += 1

        debug_print(f"ading local {name.literal} at depth {self.scope_depth} with ConstantIndex {self.num_locals - 1}")

        assert len([x for x in self.locals if x is not None]) == self.num_locals

    @EntryExit("Compiler.declare_variable")
    def declare_variable(self, name: Token):
        if self.scope_depth == 0:
            return

        for idx in range(self.num_locals - 1, -1, -1):
            local = self.locals[idx]

            assert len([x for x in self.locals if x is not None]) == self.num_locals
            assert local is not None

            if local.depth != -1 and local.depth < self.scope_depth:
                break

            if name.literal == local.name.literal:
                raise CompilerError("a variable exists with same name in this scope")

        self.add_local(name)

    @EntryExit("Compiler.mark_initialized")
    def mark_initialized(self) -> None:
        if self.scope_depth == 0:
            return

        local = self.locals[self.num_locals - 1]
        assert local is not None
        local.depth = self.scope_depth

    @EntryExit("Compiler.define_variable")
    def define_variable(self, global_index: ConstantIndex) -> None:
        if __debug__:
            assert all([_ is None for _ in self.locals[self.num_locals :]])
            debug_print(f"{self.locals[self.num_locals - 1]}, {self.scope_depth}")

        if self.scope_depth > 0:
            self.mark_initialized()
            return

        self.emit_bytes(Op.DEFINE_GLOBAL, global_index)

    @EntryExit("Compiler.resolve_local")
    def resolve_local(self, name: Token) -> None | LocalIndex:
        for idx in range(self.num_locals - 1, -1, -1):
            local = self.locals[idx]
            assert local is not None
            if local.name.literal == name.literal:
                if local.depth == -1:
                    raise CompilerError("cannot use local variable in own initializer")
                return LocalIndex(idx)

        return None

    def emit_byte(self, op: Code) -> None:
        debug_print(f"emit: {op}")
        self.function.chunk.write(op, self.vm.parser.peek(-1).line)

    def emit_bytes(self, op1: Op, op2: Code) -> None:
        self.emit_byte(op1)
        self.emit_byte(op2)

    def emit_upvalue(self, op1: bool, op2: LocalIndex | UpvalueIndex) -> None:
        self.emit_byte(op1)
        self.emit_byte(op2)

    def make_constant(self, value: Any) -> ConstantIndex:
        assert self.vm.compiler is not None
        constant_index = self.vm.compiler.function.chunk.add_constant(value)
        if constant_index > UINT8_MAX:
            raise CompilerError("too many constants in one chunk")

        debug_print(f"made constant {constant_index} -> {value} ")
        return constant_index

    @EntryExit("Compiler.identifier_constant")
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

    @EntryExit("Compiler.patch_jump")
    def patch_jump(self, offset: JumpDistance):
        self.function.chunk.code[offset] = JumpDistance(len(self.function.chunk.code) - offset - 1)
