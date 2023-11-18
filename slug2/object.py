# from enum import Enum, auto
from typing import Any, Callable, NewType

from slug2.chunk import Chunk

# class ObjType(Enum):
#     BOUND_METHOD = auto()
#     CLASS = auto()
#     CLOSURE = auto()
#     FUNCTION = auto()
#     INSTANCE = auto()
#     NATIVE = auto()
#     STRING = auto()
#     UPVALUE = auto()


# class Obj:
#     __slots__ = ("objtype", "next", "marked")

#     def __init__(self, objtype: ObjType, next: "Obj" | None):
#         self.objtype = objtype
#         self.next = next
#         self.marked: bool = False


class ObjFunction:
    def __init__(self) -> None:
        self.arity: int = 0
        self.name: str = ""
        self.chunk = Chunk()

    def __repr__(self) -> str:
        return f"<func :name {self.name} :arity {self.arity}>"


NumArgs = NewType("NumArgs", int)
Args = NewType("Args", list[Any])
NativeFn = Callable[[NumArgs, Args], Any]


# class ObjNative:
#     def __init__(self, function: NativeFn) -> None:
#         self.function = function
