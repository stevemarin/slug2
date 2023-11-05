from enum import IntEnum, auto
from typing import Any, Callable, NewType, cast

from slug2.chunk import Op
from slug2.common import ConstantIndex
from slug2.token import Token, TokenType, tokenize


class ParseError(Exception):
    pass


class Precedence(IntEnum):
    NONE = auto()
    ASSIGNMENT = auto()
    OR = auto()
    AND = auto()
    EQUALITY = auto()
    COMPARISON = auto()
    TERM = auto()
    FACTOR = auto()
    UNARY = auto()
    CALL = auto()
    PRIMARY = auto()

    def next_precedence(self) -> "Precedence":
        value = self.value + 1
        for pt in Precedence:
            if pt.value == value:
                return pt
        raise ParseError("unknown precedence")


class Parser:
    __slots__ = ("tokens", "current_index", "had_error", "panic_mode")

    def __init__(self, source: str | None = None) -> None:
        self.tokens: list[Token] = [] if source is None else tokenize(source)
        self.current_index: int = -1
        self.had_error: bool = False
        self.panic_mode: bool = False

        if __debug__:
            print("Tokens:")
            for t in self.tokens:
                print(f"    {t}")
            print()

    def current(self) -> Token:
        return self.tokens[self.current_index]

    def previous(self) -> Token:
        if not self.current_index > 0:
            raise ParseError("trying to get negative index token")
        return self.tokens[self.current_index - 1]

    def check(self, tokentype: TokenType) -> bool:
        return self.current().tokentype == tokentype

    def match(self, type: TokenType) -> bool:
        if self.check(type):
            self.current_index += 1
            return True
        return False

    def consume(self, tokentype: TokenType, msg: str) -> None:
        if not self.match(tokentype):
            raise ParseError(msg)

    def declaration(self) -> None:
        self.statement()

    def expression_statement(self) -> None:
        self.expression()

    def statement(self) -> None:
        if self.match(TokenType.ASSERT):
            self.assert_()
        else:
            self.expression_statement()

    def assert_(self) -> None:
        self.expression()
        emit_byte(Op.ASSERT, self.previous().line)

    def parse_precedence(self, precedence: Precedence) -> None:
        self.current_index += 1

        assignable = cast(Assignable, precedence <= Precedence.ASSIGNMENT)

        prefix = ParseRules[self.previous().tokentype].prefix
        if prefix is None:
            raise ParseError("expect expression")
        else:
            prefix(self, assignable)

        while precedence <= ParseRules[self.current().tokentype].precedence:
            self.current_index += 1

            infix = ParseRules[self.previous().tokentype].infix
            if infix is None:
                raise ParseError("infix func is None")
            else:
                infix(self, assignable)

        if assignable and self.match(TokenType.EQUAL):
            raise ParseError("invalid assignment target")

    def expression(self) -> None:
        self.parse_precedence(Precedence.ASSIGNMENT)


def emit_byte(op: Op | ConstantIndex, line: int) -> None:
    from slug2 import vm

    chunk = vm.current_chunk()

    if chunk is None:
        raise ParseError("current_chunk is None")

    chunk.write(op, line)


def emit_bytes(op1: Op, op2: Op | ConstantIndex, line1: int, line2: int) -> None:
    emit_byte(op1, line1)
    emit_byte(op2, line2)


def make_constant(value: Any) -> int:
    from slug2 import vm

    chunk = vm.current_chunk()
    if chunk is not None:
        return chunk.add_constant(value)
    raise RuntimeError("chunk is None for constant")


def emit_constant(value: Any, line: int) -> None:
    constant_index = make_constant(value)
    emit_bytes(Op.CONSTANT, constant_index, line, line)


def emit_return(line: int) -> None:
    emit_bytes(Op.NOOP, Op.RETURN, line, line)


def binary(parser: Parser, _: bool) -> None:
    previous = parser.previous()
    operator_type: TokenType = previous.tokentype
    line = previous.line
    parse_rule = ParseRules[operator_type]
    parser.parse_precedence(parse_rule.precedence.next_precedence())

    if __debug__:
        print(f"binary: {previous} on line {line}")

    match operator_type:
        case TokenType.PLUS:
            emit_byte(Op.ADD, line)
        case TokenType.MINUS:
            emit_byte(Op.SUBTRACT, line)
        case TokenType.STAR:
            emit_byte(Op.MULTIPLY, line)
        case TokenType.SLASH:
            emit_byte(Op.DIVIDE, line)
        case TokenType.LESS:
            emit_byte(Op.LESS, line)
        case TokenType.LESS_EQUAL:
            emit_byte(Op.LESS_EQUAL, line)
        case TokenType.GREATER:
            emit_byte(Op.GREATER, line)
        case TokenType.GREATER_EQUAL:
            emit_byte(Op.GREATER_EQUAL, line)
        case TokenType.EQUAL_EQUAL:
            emit_byte(Op.VALUE_EQUAL, line)
        case TokenType.NOT_EQUAL:
            emit_byte(Op.NOT_VALUE_EQUAL, line)
        case _:
            raise ValueError("unknown binary operator")


def unary(parser: Parser, _: bool) -> None:
    previous = parser.previous()
    operator_type: TokenType = previous.tokentype
    line = previous.line

    if __debug__:
        print(f"unary: {previous} on line {line}")

    parser.parse_precedence(Precedence.UNARY)

    match operator_type:
        # case TokenType.BANG:
        #     emit_byte(OP_NOT, line)
        case TokenType.MINUS:
            emit_byte(Op.NEGATE, line)
        case _:
            raise ValueError("unknown unary operator")


def integer(parser: Parser, _: bool) -> None:
    previous = parser.previous()
    emit_constant(previous.value, previous.line)

    if __debug__:
        print(f"integer: {previous} on line {previous.line}")


def float_(parser: Parser, _: bool) -> None:
    previous = parser.previous()
    emit_constant(previous.value, previous.line)

    if __debug__:
        print(f"float: {previous} on line {previous.line}")


def complex_(parser: Parser, _: bool) -> None:
    previous = parser.previous()
    emit_constant(previous.value, previous.line)

    if __debug__:
        print(f"complex: {previous} on line {previous.line}")


def grouping(parser: Parser, _: bool) -> None:
    parser.expression()
    parser.consume(TokenType.RIGHT_PAREN, "iaefnieafn")


def call(parser: Parser, _: bool) -> None:
    raise NotImplementedError


Assignable = NewType("Assignable", bool)
ParseFn = Callable[[Parser, Assignable], None]


class ParseRule:
    __slots__ = ("prefix", "infix", "precedence")

    def __init__(self, prefix: None | ParseFn, infix: None | ParseFn, precedence: Precedence):
        self.prefix = prefix
        self.infix = infix
        self.precedence = precedence

    def __repr__(self) -> str:
        return "<ParseRule ...>"


# fmt: off
ParseRules = {
    TokenType.LEFT_PAREN:    ParseRule(grouping,  call,   Precedence.CALL       ),
    TokenType.RIGHT_PAREN:   ParseRule(None,      None,   Precedence.NONE       ),

    TokenType.PLUS:          ParseRule(None,      binary, Precedence.TERM       ),
    TokenType.MINUS:         ParseRule(unary,     binary, Precedence.TERM       ),
    TokenType.STAR:          ParseRule(None,      binary, Precedence.FACTOR     ),
    TokenType.SLASH:         ParseRule(None,      binary, Precedence.FACTOR     ),
    TokenType.GREATER:       ParseRule(None,      binary, Precedence.COMPARISON ),
    TokenType.GREATER_EQUAL: ParseRule(None,      binary, Precedence.COMPARISON ),
    TokenType.LESS:          ParseRule(None,      binary, Precedence.COMPARISON ),
    TokenType.LESS_EQUAL:    ParseRule(None,      binary, Precedence.COMPARISON ),
    TokenType.EQUAL_EQUAL:   ParseRule(None,      binary, Precedence.EQUALITY   ),
    TokenType.NOT_EQUAL:     ParseRule(None,      binary, Precedence.EQUALITY   ),

    TokenType.INTEGER:       ParseRule(integer,   None,   Precedence.NONE       ),
    TokenType.FLOAT:         ParseRule(float_,    None,   Precedence.NONE       ),
    TokenType.COMPLEX:       ParseRule(complex_,  None,   Precedence.NONE       ),
    TokenType.ASSERT:        ParseRule(None,      None,   Precedence.NONE       ),

    TokenType.EOF:           ParseRule(None,      None,   Precedence.NONE       ),

}
# fmt: one
