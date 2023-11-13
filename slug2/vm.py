from collections import deque
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from slug2.chunk import Code, Op, ConstantIndex, JumpDistance
from slug2.common import  PythonNumber
from slug2.object import ObjFunction

if TYPE_CHECKING:
    from slug2.chunk import Chunk
    from slug2.compiler import Compiler


__max_frames__ = 256


class InterpretResult(Enum):
    OK = auto()
    COMPLE_ERROR = auto()
    RUNTIME_ERROR = auto()


class CallFrame:
    __slots__ = ("function", "ip", "slots")

    def __init__(self, function: ObjFunction, ip: list[Code], slots: int) -> None:
        self.function = function
        self.ip = ip
        self.slots = slots


class VM:
    __slots__ = ("stack", "globals", "strings", "objects", "compilers", "frames")

    def __init__(self) -> None:
        self.stack: deque[Any] = deque()
        self.globals: dict[str, Any] = {}
        self.strings: dict[str, str] = {}
        self.objects: deque[Any] = deque()
        self.compilers: deque["Compiler"] = deque()
        self.frames: deque["CallFrame"] = deque()

    def current_compiler(self) -> "Compiler | None":
        return self.compilers[-1] if len(self.compilers) > 0 else None

    def current_chunk(self) -> "Chunk | None":
        current_compiler = self.current_compiler()
        return current_compiler.function.chunk if current_compiler else None

    def peek(self, distance: int) -> Any:
        return self.stack[-1 - distance]

    def call(self, function: ObjFunction, num_args: int) -> bool:
        print("function.name", function.name)
        if num_args != function.arity:
            raise RuntimeError(f"expected {function.arity} arguments: got {num_args}")

        if len(self.frames) == __max_frames__:
            raise RuntimeError("stack overflow")

        frame = CallFrame(function, function.chunk.code, len(self.stack) - num_args - 1)
        self.frames.append(frame)

        return True

    def run(self) -> InterpretResult:
        frame = self.frames[-1]
        ip_index: int = 0

        def read_byte() -> tuple[int, Code]:
            return ip_index + 1, frame.ip[ip_index]

        def read_constant() -> tuple[int, Any]:
            ip_index, value_index = read_byte()
            if isinstance(value_index, ConstantIndex):
                return ip_index, frame.function.chunk.constants[value_index]
            else:
                raise TypeError(f"expected ConstantIndex, got {type(value_index)}")

        def binary_op(op: Op) -> None:
            right: PythonNumber = self.stack.pop()
            left: PythonNumber = self.stack.pop()
            self.stack.append(op.evaluate_binary(left, right))

        while len(self.stack) > 0:
            print("stack", self.stack)

            ip_index, instruction = read_byte()

            # if type(instruction) == ConstantIndex:
            #     instruction = frame.function.chunk.constants[instruction]

            if not isinstance(instruction, Op):
                raise RuntimeError(f"instruction is {type(instruction)} not Op")

            match instruction:
                case Op.CONSTANT:
                    ip_index, constant = read_constant()
                    self.stack.append(constant)
                case Op.TRUE:
                    self.stack.append(True)
                case Op.FALSE:
                    self.stack.append(False)
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
                # case Op.NOOP:
                #     pass
                case Op.ASSERT:
                    test = self.stack.pop()
                    if not isinstance(test, bool):
                        raise RuntimeError("assert only works with bools")
                    if not test:
                        raise AssertionError("assert failed")
                    self.stack.append(Op.NOOP)
                case Op.PRINT:
                    print(f"Printing from Slug2: {self.stack.pop()}")
                case Op.JUMP:
                    ip_index, instruction = read_byte()
                    if not isinstance(instruction, JumpDistance):
                        raise RuntimeError(f"expected JumpDistance, got {type(instruction)}")
                    ip_index += int(instruction)
                case Op.JUMP_IF_FALSE:
                    ip_index, instruction = read_byte()
                    if not isinstance(instruction, JumpDistance):
                        raise RuntimeError(f"expected JumpDistance, got {type(instruction)}")
                    if self.peek(0) is False:
                        ip_index += int(instruction)
                case Op.RETURN:
                    result = self.stack.pop()
                    _ = self.frames.pop()
                    if len(self.frames) == 0:
                        # _ = self.stack.pop()
                        return InterpretResult.OK

                    self.stack.append(result)
                    frame = self.frames[-1]
                case _:
                    raise NotImplementedError(f"op {instruction} not implemented")

        return InterpretResult.OK

    def interpret(self, source: str) -> InterpretResult:
        maybe_func = self.compile(source)
        if maybe_func is None:
            return InterpretResult.COMPLE_ERROR

        maybe_func.name = "SCRIPT"
        self.stack.append(maybe_func)
        self.call(maybe_func, 0)

        # if __debug__:
        #     print()
        #     print("Ops:")
        #     for op in self.compilers[-1].function.chunk.code:
        #         if type(op) == ConstantIndex:
        #             print(f"    {op} -> {self.compilers[-1].function.chunk.constants[op]}")
        #         elif type(op) == JumpDistance:
        #             print(f"    {op} jump to {self.compilers[-1].function.chunk.constants[op]}")
        #         else:
        #             print(f"    {op}")
        #     print()

        return self.run()

    def compile(self, source: str) -> ObjFunction | None:
        from slug2.compiler import Compiler, FuncType

        root_compiler = Compiler(source, None, FuncType.SCRIPT)
        self.compilers.append(root_compiler)
        maybe_func = root_compiler.compile()

        return maybe_func
