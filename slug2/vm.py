from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from slug2.chunk import Code, Op
from slug2.common import PythonNumber, ConstantIndex, LocalIndex, JumpDistance
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
    __slots__ = ("function", "ip", "slots_start")

    def __init__(self, function: ObjFunction, ip: list[Code], slots_start: int) -> None:
        self.function = function
        self.ip = ip
        self.slots_start: int = slots_start


class VM:
    __slots__ = ("stack", "globals", "strings", "objects", "compilers", "frames")

    def __init__(self) -> None:
        self.stack: list[Any] = list()
        self.globals: dict[str, Any] = {}
        self.strings: dict[str, str] = {}
        self.objects: list[Any] = list()
        self.compilers: list["Compiler"] = list()
        self.frames: list["CallFrame"] = list()

    def current_compiler(self) -> "Compiler | None":
        return self.compilers[-1] if len(self.compilers) > 0 else None

    def current_chunk(self) -> "Chunk | None":
        current_compiler = self.current_compiler()
        return current_compiler.function.chunk if current_compiler else None

    def peek(self, distance: int = 0) -> Any:
        return self.stack[-1 - distance]

    def call(self, function: ObjFunction, num_args: int) -> bool:
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
            ip_index, instruction = read_byte()
            
            if __debug__:
                print(f"\nSTACK :inst {instruction}")
                for item in self.stack:
                    print(f"  {type(item)}: {item}")
                print()

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
                case Op.SET_LOCAL:
                    ip_index, slot = read_byte()
                    if not isinstance(slot, LocalIndex):
                        raise RuntimeError("expected SlotInstance")
                    self.stack[frame.slots_start + slot] = self.peek(0)
                case Op.GET_LOCAL:
                    ip_index, slot = read_byte()
                    if not isinstance(slot, LocalIndex):
                        raise RuntimeError("expected SlotInstance")
                    self.stack.append(self.stack[frame.slots_start + slot])
                case Op.DEFINE_GLOBAL:
                    ip_index, name = read_constant()
                    if not isinstance(name, str):
                        raise RuntimeError("global name not string")
                    self.globals[name] = self.peek()
                    _ = self.stack.pop()
                case Op.SET_GLOBAL:
                    ip_index, name = read_constant()
                    if not isinstance(name, str):
                        raise RuntimeError("global name not string")
                    if name not in self.globals:
                        raise RuntimeError(f"global variable {name} not defined")
                    self.globals[name] = self.peek()
                case Op.GET_GLOBAL:
                    ip_index, name = read_constant()
                    if not isinstance(name, str):
                        raise RuntimeError("global name not string")
                    if name not in self.globals:
                        raise RuntimeError(f"global variable {name} not defined")
                    self.stack.append(self.globals[name])
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
                    if not isinstance(self.peek(0), bool):
                        raise RuntimeError("only booleans are truthy")
                    if self.peek() is False:
                        ip_index += int(instruction)
                case Op.RETURN:
                    result = self.stack.pop()
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
        from slug2.compiler import Compiler, FuncType

        root_compiler = Compiler(source, None, FuncType.SCRIPT)
        self.compilers.append(root_compiler)
        maybe_func = root_compiler.compile()
        _ = self.compilers.pop()

        return maybe_func

    def interpret(self, source: str) -> InterpretResult:
        maybe_func = self.compile(source)
        if maybe_func is None:
            return InterpretResult.COMPLE_ERROR
        
        print(maybe_func.chunk)
        print(maybe_func.chunk.constants)
        from slug2 import vm
        print(vm.frames)

        maybe_func.name = "SCRIPT"
        self.stack.append(maybe_func)
        self.call(maybe_func, 0)

        return self.run()

    def __repr__(self) -> str:
        repr = "\nStack:\n"
        for code in self.stack:
            repr += str(code) + "\n"
        return repr
