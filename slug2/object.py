# from enum import Enum, auto
from abc import ABC
from enum import Enum, auto
from typing import TYPE_CHECKING

from slug2.chunk import Chunk
from slug2.common import FuncType, StackIndex

if TYPE_CHECKING:
    from slug2.vm import VM


class ObjType(Enum):
    BOUND_METHOD = auto()
    CLASS = auto()
    CLOSURE = auto()
    FUNCTION = auto()
    INSTANCE = auto()
    NATIVE = auto()
    STRING = auto()
    UPVALUE = auto()


class Obj(ABC):
    __slots__ = ("marked", "objtype", "next")

    def __init__(self, vm: "VM", objtype: ObjType):
        self.marked: bool = False
        self.objtype: ObjType = objtype
        self.next = vm.objects
        vm.objects = self


class ObjFunction(Obj):
    def __init__(self, vm: "VM", objtype: ObjType, functype: FuncType) -> None:
        super().__init__(vm, objtype)
        self.arity: int = 0
        self.upvalue_count: int = 0
        self.name: str = ""
        self.chunk = Chunk()
        self.functype: FuncType = functype

    def __repr__(self) -> str:
        return f"<ObjFunction :name {self.name} :arity {self.arity}>"


class ObjUpvalue(Obj):
    def __init__(self, vm: "VM", objtype: ObjType, stack_index: StackIndex):
        super().__init__(vm, objtype)
        self.closed = None
        self.stack_index: StackIndex | None = stack_index
        self.next_upvalue: "ObjUpvalue | None" = None

    def __repr__(self) -> str:
        return f"<ObjUpvalue :closed {self.closed} :location {self.stack_index}>"


class ObjClosure(Obj):
    def __init__(self, vm: "VM", objtype: ObjType, function: ObjFunction):
        super().__init__(vm, objtype)
        self.function = function
        self.upvalue_count = function.upvalue_count
        self.upvalues: list["ObjUpvalue | None"] = [None] * function.upvalue_count

    def __repr__(self) -> str:
        return f"<ObjClosure :function {self.function}>"


# NumArgs = NewType("NumArgs", int)
# Args = NewType("Args", list[Any])
# NativeFn = Callable[[NumArgs, Args], Any]


# class ObjNative:
#     def __init__(self, function: NativeFn) -> None:
#         self.function = function
