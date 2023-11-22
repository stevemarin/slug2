from enum import Enum, auto
from textwrap import dedent
from typing import TYPE_CHECKING, Any

from slug2.common import (
    FRAMES_MAX,
    Code,
    ConstantIndex,
    FuncType,
    JumpDistance,
    LocalIndex,
    NumArgs,
    Op,
    PythonNumber,
    StackIndex,
    UpvalueIndex,
)
from slug2.compiler import Compiler
from slug2.object import ObjClosure, ObjFunction, ObjType, ObjUpvalue
from slug2.parser import Parser

if TYPE_CHECKING:
    from slug2.object import Obj


class InterpretResult(Enum):
    OK = auto()
    COMPLE_ERROR = auto()
    RUNTIME_ERROR = auto()


class CallFrame:
    __slots__ = ("closure", "instructions", "ip", "slots_start")

    def __init__(self, closure: ObjClosure, instructions: list[Code], slots_start: StackIndex) -> None:
        self.closure = closure
        self.instructions = instructions
        self.ip = 0
        self.slots_start: StackIndex = slots_start

    def read_byte(self) -> Code:
        code = self.instructions[self.ip]
        self.ip += 1
        return code

    def read_constant(self) -> Any:
        constant_index = self.read_byte()
        if isinstance(constant_index, ConstantIndex):
            return self.closure.function.chunk.constants[constant_index]
        else:
            raise TypeError(f"expected ConstantIndex, got {type(constant_index)}")

    def __repr__(self) -> str:
        newline = "\n"
        return dedent(
            f"""
            <Frame 
              :closure {self.closure} 
              :ip {str(self.closure.function.chunk).strip().replace(newline, newline + "                  ")} 
              :slots_start {self.slots_start}>"""
        )


class VM:
    __slots__ = (
        "frames",
        "stack",
        "globals",
        "strings",
        "init_string",
        "open_upvalues",
        "objects",
        "parser",
        "compiler",
    )

    def __init__(self) -> None:
        self.frames: list["CallFrame"] = list()
        self.stack: list[Any] = list()
        self.globals: dict[str, Any] = {}
        self.strings: dict[str, str] = {}
        self.init_string = "init"
        self.open_upvalues: "ObjUpvalue | None" = None
        self.objects: "Obj | None" = None
        self.parser: Parser = Parser(self)
        self.compiler: Compiler = Compiler(self, "SCRIPT", FuncType.SCRIPT)

    def peek(self, distance: int = 0) -> Any:
        return self.stack[-1 - distance]

    def call(self, closure: ObjClosure, num_args: int) -> bool:
        if num_args != closure.function.arity:
            raise RuntimeError(f"expected {closure.function.arity} arguments: got {num_args}")

        if len(self.frames) == FRAMES_MAX:
            raise RuntimeError("stack overflow")

        frame = CallFrame(closure, closure.function.chunk.code, StackIndex(len(self.stack) - num_args - 1))
        self.frames.append(frame)

        return True

    def call_value(self, num_args: int) -> bool:
        callee: Any = self.peek(num_args)
        print("AAA", callee, num_args, type(callee))
        match callee.objtype:
            case ObjType.BOUND_METHOD:
                raise NotImplementedError
            case ObjType.CLASS:
                raise NotImplementedError
            case ObjType.CLOSURE:
                return self.call(callee, num_args)
            case ObjType.NATIVE:
                raise NotImplementedError
            case _:
                raise RuntimeError("can only call functions and classes")

    def capture_upvalue(self, local: Any) -> ObjUpvalue:
        prev_upvalue: ObjUpvalue | None = None
        upvalue: ObjUpvalue | None = self.open_upvalues
        while upvalue is not None and upvalue.stack_index > local:
            prev_upvalue = upvalue
            if isinstance(upvalue.next, ObjUpvalue):
                upvalue = upvalue.next
            else:
                raise RuntimeError(f"expected ObjUpvalue, got {type(upvalue.next)}")

        if upvalue is not None and upvalue.stack_index == local:
            return upvalue

        created_upvalue = ObjUpvalue(self, ObjType.CLOSURE, local)
        created_upvalue.next = upvalue

        if prev_upvalue is None:
            self.open_upvalues = created_upvalue
        else:
            prev_upvalue.next = created_upvalue

        return created_upvalue

    def close_upvalue(self, last: StackIndex) -> None:
        while True:
            upvalue = self.open_upvalues

            if upvalue is None:
                return
            elif not isinstance(upvalue, ObjUpvalue):
                raise RuntimeError("not an ObjUpvalue")
            elif upvalue.stack_index is None:
                raise RuntimeError("already closed")
            elif upvalue.stack_index >= last:
                return
            else:
                upvalue.closed = self.stack[upvalue.stack_index]
                upvalue.stack_index = None
                self.open_upvalues = upvalue.next_upvalue

    def run(self) -> InterpretResult:
        frame = self.frames[-1]

        def binary_op(op: Op) -> None:
            right: PythonNumber = self.stack.pop()
            left: PythonNumber = self.stack.pop()
            self.stack.append(op.evaluate_binary(left, right))

        while len(self.stack) > 0:
            instruction = frame.read_byte()

            if __debug__:
                print(f"\nSTACK :inst {instruction} :depth {len(self.frames)}")
                for item in self.stack:
                    print(f"  {type(item)}: {item}")
                print()

            if not isinstance(instruction, Op):
                raise RuntimeError(f"instruction is {type(instruction)} not Op")

            match instruction:
                case Op.CONSTANT:
                    self.stack.append(frame.read_constant())
                case Op.TRUE:
                    self.stack.append(True)
                case Op.FALSE:
                    self.stack.append(False)
                case Op.NONE:
                    self.stack.append(None)
                case Op.POP:
                    self.stack.pop()
                case (
                    Op.ADD
                    | Op.SUBTRACT
                    | Op.MULTIPLY
                    | Op.DIVIDE
                    | Op.EXPONENT
                    | Op.LESS
                    | Op.LESS_EQUAL
                    | Op.GREATER
                    | Op.GREATER_EQUAL
                    | Op.VALUE_EQUAL
                    | Op.NOT_VALUE_EQUAL
                ):
                    binary_op(instruction)
                case Op.NEGATE:
                    self.stack[-1] *= -1
                case Op.SET_LOCAL:
                    slot = frame.read_byte()
                    if not isinstance(slot, LocalIndex):
                        raise RuntimeError("expected SlotInstance")
                    self.stack[frame.slots_start + slot] = self.peek(0)
                case Op.GET_LOCAL:
                    slot = frame.read_byte()
                    if not isinstance(slot, LocalIndex):
                        raise RuntimeError("expected SlotInstance")
                    self.stack.append(self.stack[frame.slots_start + slot])
                case Op.DEFINE_GLOBAL:
                    name = frame.read_constant()
                    if not isinstance(name, str):
                        raise RuntimeError("global name not string")
                    self.globals[name] = self.peek()
                    _ = self.stack.pop()
                case Op.SET_GLOBAL:
                    name = frame.read_constant()
                    if not isinstance(name, str):
                        raise RuntimeError("global name not string")
                    if name not in self.globals:
                        raise RuntimeError(f"global variable {name} not defined")
                    self.globals[name] = self.peek()
                case Op.GET_GLOBAL:
                    name = frame.read_constant()
                    if not isinstance(name, str):
                        raise RuntimeError("global name not string")
                    if name not in self.globals:
                        raise RuntimeError(f"global variable {name} not defined")
                    self.stack.append(self.globals[name])
                case Op.SET_UPVALUE:
                    upvalue_index = frame.read_byte()
                    if not isinstance(upvalue_index, UpvalueIndex):
                        raise RuntimeError(f"expexted UpvalueIndex, got:: {type(upvalue_index)}")
                    upvalue = frame.closure.upvalues[upvalue_index]
                    if not isinstance(upvalue, ObjUpvalue):
                        raise RuntimeError(f"expected ObjUpvalue, got {type(upvalue)}")
                    upvalue.stack_index = self.peek()
                case Op.GET_UPVALUE:
                    upvalue_index = frame.read_byte()
                    if not isinstance(upvalue_index, UpvalueIndex):
                        raise RuntimeError(f"expexted UpvalueIndex, got:: {type(upvalue_index)}")
                    upvalue = frame.closure.upvalues[upvalue_index]
                    if not isinstance(upvalue, ObjUpvalue):
                        raise RuntimeError(f"expected ObjUpvalue, got {type(upvalue)}")
                    self.stack.append(upvalue.stack_index)
                case Op.ASSERT:
                    test = self.stack.pop()
                    if not isinstance(test, bool):
                        raise RuntimeError("assert only works with bools")
                    if not test:
                        raise AssertionError("assert failed")
                case Op.PRINT:
                    print(f"Printing from Slug2: {self.stack.pop()}")
                case Op.JUMP:
                    instruction = frame.read_byte()
                    if not isinstance(instruction, JumpDistance):
                        raise RuntimeError(f"expected JumpDistance, got {type(instruction)}")
                    frame.ip += int(instruction)
                case Op.JUMP_IF_FALSE:
                    instruction = frame.read_byte()
                    if not isinstance(instruction, JumpDistance):
                        raise RuntimeError(f"expected JumpDistance, got {type(instruction)}")
                    if not isinstance(self.peek(0), bool):
                        raise RuntimeError("only booleans are truthy")
                    if self.peek() is False:
                        frame.ip += int(instruction)
                case Op.CALL:
                    num_args = frame.read_byte()
                    if not isinstance(num_args, NumArgs):
                        raise RuntimeError(f"expected NumArgs. got {type(num_args)}")
                    if not self.call_value(num_args):
                        return InterpretResult.RUNTIME_ERROR
                    frame = self.frames[-1]
                case Op.CLOSURE:
                    function = frame.read_constant()
                    if not isinstance(function, ObjFunction):
                        raise RuntimeError(f"expected ObjFunction, got {type(function)}")

                    closure = ObjClosure(self, ObjType.CLOSURE, function)
                    self.stack.append(closure)
                    for idx in range(closure.function.num_upvalues):
                        is_local = frame.read_byte()
                        if not isinstance(is_local, bool):
                            raise RuntimeError(f"expected bool, got {type(is_local)}, {is_local}")

                        index = frame.read_byte()
                        if not isinstance(index, LocalIndex | UpvalueIndex):
                            raise RuntimeError(f"expected ConstantIndex, got {type(index)}")

                        if is_local:
                            closure.upvalues[idx] = self.capture_upvalue(frame.slots_start + index)
                        else:
                            closure.upvalues[idx] = frame.closure.upvalues[index]
                case Op.CLOSE_UPVALUE:
                    self.close_upvalue(StackIndex(len(self.stack) - 1))
                    self.stack.pop()
                case Op.RETURN:
                    result = self.stack.pop()
                    self.close_upvalue(frame.slots_start)
                    _ = self.frames.pop()
                    if len(self.frames) == 0:
                        _ = self.stack.pop()
                        return InterpretResult.OK

                    self.stack.append(result)
                    frame = self.frames[-1]
                case _:
                    raise NotImplementedError(f"op {instruction} not implemented")

        return InterpretResult.OK

    def compile(self, source: str) -> ObjFunction | None:
        self.parser = Parser(self, source)
        # self.compiler = Compiler(self, FuncType.SCRIPT)
        maybe_func = self.compiler.compile()
        # self.compiler = self.compiler.enclosing

        return maybe_func

    def interpret(self, source: str) -> InterpretResult:
        maybe_func = self.compile(source)
        if maybe_func is None:
            return InterpretResult.COMPLE_ERROR
        closure = ObjClosure(self, ObjType.CLOSURE, maybe_func)

        print(closure.function.chunk)
        print(closure.function.chunk.constants)
        print(closure.function)
        print(self.frames)

        closure.function.name = "SCRIPT"
        self.stack.append(closure)
        self.call(closure, 0)

        return self.run()

    def __repr__(self) -> str:
        repr = "\nStack:\n"
        for code in self.stack:
            repr += str(code) + "\n"
        return repr
