from enum import Enum, auto
from textwrap import dedent
from typing import TYPE_CHECKING, Any

from slug2.common import (
    FRAMES_MAX,
    UINT8_MAX,
    Code,
    ConstantIndex,
    FuncType,
    JumpDistance,
    LocalIndex,
    NumArgs,
    Op,
    PythonNumber,
    StackIndex,
    Uninitialized,
    UpvalueIndex,
    debug_print,
    uninitialized,
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
        self.slots_start: StackIndex = slots_start
        self.ip = 0

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
        "num_frames",
        "stack",
        "stack_top",
        "globals",
        "strings",
        "init_string",
        "open_upvalue",
        "objects",
        "parser",
        "compiler",
    )

    def __init__(self) -> None:
        self.frames: list["CallFrame | Uninitialized"] = [uninitialized] * UINT8_MAX
        self.num_frames: int = 0

        self.stack: list[Any] = [uninitialized] * UINT8_MAX
        self.stack_top: int = 0

        self.globals: dict[str, Any] = {}
        self.strings: dict[str, str] = {}
        self.init_string = "init"
        self.open_upvalue: "ObjUpvalue | None" = None
        self.objects: "Obj | None" = None
        self.parser: Parser = Parser(self)
        self.compiler: "Compiler | None" = None

    def push(self, value: Any) -> None:
        assert not isinstance(value, Uninitialized)
        self.stack[self.stack_top] = value
        self.stack_top += 1

        print("pushing:", value)

    def pop(self) -> Any:
        self.stack_top -= 1
        value = self.stack[self.stack_top]
        assert not isinstance(value, Uninitialized)

        self.stack[self.stack_top] = uninitialized

        print("popping:", value)

        return value

    def peek(self, distance: int = 0) -> Any:
        return self.stack[self.stack_top - 1 - distance]

    def call(self, closure: ObjClosure, num_args: int) -> bool:
        if num_args != closure.function.arity:
            print("aaa", closure.function)
            print("bbb", self)
            raise RuntimeError(f"expected {closure.function.arity} arguments: got {num_args}")

        if self.num_frames == FRAMES_MAX:
            raise RuntimeError("stack overflow")

        slots = StackIndex(self.stack_top - num_args - 1)
        frame = CallFrame(closure, closure.function.chunk.code, slots)
        self.frames[self.num_frames] = frame
        self.num_frames += 1

        return True

    def call_value(self, num_args: int) -> bool:
        callee: Any = self.peek(num_args)
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
        upvalue: ObjUpvalue | None = self.open_upvalue
        while upvalue is not None and upvalue.stack_index > local:
            prev_upvalue = upvalue
            upvalue = upvalue.next

        if upvalue is not None and upvalue.stack_index == local:
            return upvalue

        created_upvalue = ObjUpvalue(self, ObjType.CLOSURE, local)
        created_upvalue.next = upvalue

        if prev_upvalue is None:
            self.open_upvalue = created_upvalue
        else:
            prev_upvalue.next = created_upvalue

        return created_upvalue

    def close_upvalue(self, last: StackIndex) -> None:
        while self.open_upvalue is not None and self.open_upvalue.stack_index >= last:
            upvalue = self.open_upvalue
            upvalue.closed = upvalue.stack_index
            self.open_upvalue = upvalue.next

    def run(self) -> InterpretResult:
        frame = self.frames[self.num_frames - 1]

        def binary_op(op: Op) -> None:
            right: PythonNumber = self.pop()
            left: PythonNumber = self.pop()
            self.push(op.evaluate_binary(left, right))

        while True:
            assert not isinstance(frame, Uninitialized)
            if frame.ip >= len(frame.instructions):
                return InterpretResult.OK

            instruction = frame.read_byte()

            if __debug__:
                print(instruction)
                print(self)

            if not isinstance(instruction, Op):
                raise RuntimeError(f"instruction is {type(instruction)} not Op")

            match instruction:
                case Op.CONSTANT:
                    self.push(frame.read_constant())
                case Op.TRUE:
                    self.push(True)
                case Op.FALSE:
                    self.push(False)
                case Op.NONE:
                    self.push(None)
                case Op.POP:
                    self.pop()
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
                    self.stack[self.stack_top - 1] *= -1
                case Op.SET_LOCAL:
                    slot = frame.read_byte()
                    if not isinstance(slot, LocalIndex):
                        raise RuntimeError("expected SlotInstance")
                    self.stack[frame.slots_start + slot] = self.peek(0)
                case Op.GET_LOCAL:
                    slot = frame.read_byte()
                    if not isinstance(slot, LocalIndex):
                        raise RuntimeError("expected SlotInstance")
                    self.push(self.stack[frame.slots_start + slot])
                case Op.DEFINE_GLOBAL:
                    name = frame.read_constant()
                    if not isinstance(name, str):
                        raise RuntimeError("global name not string")
                    self.globals[name] = self.peek()
                    _ = self.pop()
                case Op.SET_GLOBAL:
                    name = frame.read_constant()
                    assert isinstance(name, str)
                    if name not in self.globals:
                        raise RuntimeError(f"global variable {name} not defined")
                    self.globals[name] = self.peek()
                    print("SET GLOBALS:", self.globals)
                case Op.GET_GLOBAL:
                    name = frame.read_constant()
                    assert isinstance(name, str)
                    if name not in self.globals:
                        raise RuntimeError(f"global variable {name} not defined")
                    print("GET GLOBALS:", self.globals)
                    self.push(self.globals[name])
                case Op.SET_UPVALUE:
                    upvalue_index = frame.read_byte()
                    assert isinstance(upvalue_index, UpvalueIndex), type(upvalue_index)
                    upvalue = frame.closure.upvalues[upvalue_index]
                    assert isinstance(upvalue, ObjUpvalue), type(upvalue)
                    upvalue.stack_index = self.peek()
                case Op.GET_UPVALUE:
                    upvalue_index = frame.read_byte()
                    assert isinstance(upvalue_index, UpvalueIndex), type(upvalue_index)
                    assert 0 <= upvalue_index <= frame.closure.num_upvalues, upvalue_index
                    debug_print(f"{frame.closure.upvalues}, {frame.closure.num_upvalues}")
                    upvalue = frame.closure.upvalues[upvalue_index]
                    assert isinstance(upvalue, ObjUpvalue), type(upvalue)
                    self.push(upvalue.stack_index)
                case Op.ASSERT:
                    test = self.pop()
                    if not isinstance(test, bool):
                        raise RuntimeError("assert only works with bools")
                    if not test:
                        raise AssertionError("assert failed")
                case Op.PRINT:
                    print(f"Printing from Slug2: {self.pop()}")
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
                    assert isinstance(num_args, NumArgs)
                    if not self.call_value(num_args):
                        return InterpretResult.RUNTIME_ERROR
                    frame = self.frames[self.num_frames - 1]
                case Op.CLOSURE:
                    function = frame.read_constant()
                    assert isinstance(function, ObjFunction)

                    closure = ObjClosure(self, ObjType.CLOSURE, function)
                    self.push(closure)

                    for idx in range(closure.num_upvalues):
                        is_local = frame.read_byte()
                        index = frame.read_byte()

                        assert isinstance(is_local, bool)
                        assert isinstance(index, LocalIndex | UpvalueIndex)

                        if is_local:
                            closure.upvalues[idx] = self.capture_upvalue(frame.slots_start + index)
                        else:
                            closure.upvalues[idx] = frame.closure.upvalues[index]
                case Op.CLOSE_UPVALUE:
                    self.close_upvalue(StackIndex(self.stack_top - 1))
                    _ = self.pop()
                case Op.RETURN:
                    result = self.pop()
                    self.close_upvalue(frame.slots_start)

                    self.frames[self.num_frames - 1] = uninitialized
                    self.num_frames -= 1

                    if self.num_frames == 0:
                        _ = self.pop()
                        return InterpretResult.OK

                    self.stack_top = frame.slots_start
                    self.push(result)

                    frame = self.frames[self.num_frames - 1]
                case _:
                    raise NotImplementedError(f"op {instruction} not implemented")

    def interpret(self, source: str) -> InterpretResult:
        self.parser = Parser(self, source)
        self.compiler = Compiler(self, FuncType.SCRIPT)

        maybe_func = self.compiler.compile()
        if maybe_func is None:
            return InterpretResult.COMPLE_ERROR

        # push pop here for gc
        self.push(maybe_func)
        closure = ObjClosure(self, ObjType.CLOSURE, maybe_func)
        _ = self.pop()

        self.push(closure)
        self.call(closure, 0)

        return self.run()

    def __repr__(self) -> str:
        max_idx = max([idx for idx, s in enumerate(self.stack) if not isinstance(s, Uninitialized)])
        repr = f"\nStack :depth {self.num_frames}\n"
        for item in self.stack[: max_idx + 1]:
            repr += str(item) + "\n"
        return repr
