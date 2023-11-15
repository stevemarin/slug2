from enum import IntEnum, auto
from typing import Any, Callable, NewType, cast

from slug2.chunk import Code, Op, Chunk, JumpDistance, ConstantIndex
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
    EXPONENT = auto()
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

    def peek(self, distance: int = 0) -> Token:
        index = self.current_index + distance
        if not 0 <= index < len(self.tokens):
            raise RuntimeError("bad index")

        return self.tokens[index]

    def check(self, tokentype: TokenType) -> bool:
        return self.tokens[self.current_index].tokentype == tokentype

    def check_multiple(self, tokentypes: set[TokenType]) -> bool:
        return self.tokens[self.current_index].tokentype in tokentypes

    def match(self, type: TokenType) -> bool:
        if self.check(type):
            self.current_index += 1
            return True
        return False

    def consume(self, tokentype: TokenType, msg: str) -> None:
        if not self.match(tokentype):
            raise ParseError(msg)

    def strip_current_newlines(self) -> None:
        strip: int = 0
        while self.peek(strip).tokentype == TokenType.NEWLINE:
            strip += 1

        self.tokens = (  # stripping the current consecutive newlines from tokens
            self.tokens[: self.current_index] + self.tokens[self.current_index + strip :]
        )

    def parse_precedence(self, precedence: Precedence, group: bool = False) -> None:
        self.current_index += 1

        if group:
            self.strip_current_newlines()

        assignable = cast(Assignable, precedence <= Precedence.ASSIGNMENT)

        prefix = ParseRules[self.peek(-1).tokentype].prefix
        if prefix is None:
            raise ParseError(f"no prefix rule for {self.peek(-1).tokentype}")
        else:
            prefix(self, assignable)

        while precedence <= ParseRules[self.peek().tokentype].precedence:
            self.current_index += 1

            if group:
                self.strip_current_newlines()

            infix = ParseRules[self.peek(-1).tokentype].infix
            if infix is None:
                raise ParseError(f"no infix rule for {self.peek(-1).tokentype}")
            else:
                infix(self, assignable)
            
            if group:
                self.strip_current_newlines()

        if assignable and self.match(TokenType.EQUAL):
            raise ParseError("invalid assignment target")

    def expression(self, group: bool = False) -> None:
        self.parse_precedence(Precedence.ASSIGNMENT, group=group)

    def expression_statement(self) -> None:
        self.expression()
        emit_byte(Op.POP, self.peek(-1).line)

    def assert_statement(self) -> None:
        self.expression()
        emit_byte(Op.ASSERT, self.peek(-1).line)

    def print_statement(self):
        self.expression()
        emit_byte(Op.PRINT, self.peek(-1).line)

    def skip_newlines(self) -> None:
        while self.peek().tokentype == TokenType.NEWLINE:
            self.current_index += 1

    def if_statement(self) -> None:
        jumps_to_end: list[JumpDistance] = []

        while True:  # in if-statement right after the if
            self.expression()
            self.consume(TokenType.NEWLINE, "expect newline after if clause")

            jump_to_next_clause = emit_jump(Op.JUMP_IF_FALSE, self.peek(-1).line)
            emit_byte(Op.POP, self.peek(-1).line)

            self.block(set([TokenType.ELSE, TokenType.END]))

            jumps_to_end.append(emit_jump(Op.JUMP, self.peek(-1).line))

            patch_jump(jump_to_next_clause)
            emit_byte(Op.POP, self.peek(-1).line)

            if self.match(TokenType.ELSE):  # consumes the else
                if self.match(TokenType.IF):  # consumes the if
                    continue  # else if statement -> back to top of loop
                else:  # bare else statement -> fall through to break
                    self.block(set([TokenType.END]))

            break

        self.consume(TokenType.END, "expect end after if block")

        for jump in jumps_to_end:
            patch_jump(jump)

    def statement(self) -> None:
        if self.match(TokenType.NEWLINE):
            pass
        elif self.match(TokenType.ASSERT):
            self.assert_statement()
        elif self.match(TokenType.PRINT):
            self.print_statement()
        elif self.match(TokenType.IF):
            self.if_statement()
        else:
            self.expression_statement()

    def declaration(self) -> None:
        self.statement()

    def block(self, end_tokentypes: set[TokenType]) -> None:
        end_tokentypes_or_eof = end_tokentypes | set([TokenType.EOF])
        while not self.check_multiple(end_tokentypes_or_eof):
            self.declaration()


def current_chunk() -> Chunk:
    from slug2 import vm

    chunk = vm.current_chunk()
    if chunk is None:
        raise RuntimeError("chunk is None")
    return chunk


def emit_byte(op: Code, line: int) -> None:
    from slug2 import vm

    chunk = vm.current_chunk()

    if chunk is None:
        raise ParseError("current_chunk is None")

    chunk.write(op, line)


def emit_bytes(op1: Op, op2: Code, line1: int, line2: int) -> None:
    emit_byte(op1, line1)
    emit_byte(op2, line2)


def make_constant(value: Any) -> ConstantIndex:
    chunk = current_chunk()
    return chunk.add_constant(value)


def emit_constant(value: Any, line: int) -> None:
    constant_index = make_constant(value)
    emit_bytes(Op.CONSTANT, constant_index, line, line)


def emit_return(line: int) -> None:
    emit_bytes(Op.TRUE, Op.RETURN, line, line)


def emit_jump(op: Op, line: int) -> JumpDistance:
    emit_byte(op, line)
    emit_byte(Op.JUMP_FAKE, line)
    return JumpDistance(len(current_chunk().code) - 1)


def patch_jump(offset: JumpDistance):
    chunk = current_chunk()
    chunk.code[offset] = JumpDistance(len(chunk.code) - offset - 1)


def binary(parser: Parser, _: bool) -> None:
    previous = parser.peek(-1)
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
        case TokenType.STAR_STAR:
            emit_byte(Op.EXPONENT, line)
        case TokenType.SLASH:
            emit_byte(Op.DIVIDE, line)
        case TokenType.SLASH_SLASH:
            emit_byte(Op.INT_DIVIDE, line)
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
    previous = parser.peek(-1)
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


def literal(parser: Parser, _: bool) -> None:
    previous = parser.peek(-1)
    line = previous.line
    match tokentype := previous.tokentype:
        case TokenType.TRUE:
            emit_byte(Op.TRUE, line)
        case TokenType.FALSE:
            emit_byte(Op.FALSE, line)
        case _:
            RuntimeError(f"unknown literal {tokentype}")


def integer(parser: Parser, _: bool) -> None:
    previous = parser.peek(-1)
    emit_constant(previous.value, previous.line)

    if __debug__:
        print(f"integer: {previous} on line {previous.line}")


def float_(parser: Parser, _: bool) -> None:
    previous = parser.peek(-1)
    emit_constant(previous.value, previous.line)

    if __debug__:
        print(f"float: {previous} on line {previous.line}")


def complex_(parser: Parser, _: bool) -> None:
    previous = parser.peek(-1)
    emit_constant(previous.value, previous.line)

    if __debug__:
        print(f"complex: {previous} on line {previous.line}")


def grouping(parser: Parser, _: bool) -> None:
    parser.expression(group=True)
    parser.consume(TokenType.RIGHT_PAREN, "didn't find closing )")


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
    TokenType.STAR_STAR:     ParseRule(None,      binary, Precedence.EXPONENT   ),
    TokenType.GREATER:       ParseRule(None,      binary, Precedence.COMPARISON ),
    TokenType.GREATER_EQUAL: ParseRule(None,      binary, Precedence.COMPARISON ),
    TokenType.LESS:          ParseRule(None,      binary, Precedence.COMPARISON ),
    TokenType.LESS_EQUAL:    ParseRule(None,      binary, Precedence.COMPARISON ),
    TokenType.EQUAL_EQUAL:   ParseRule(None,      binary, Precedence.EQUALITY   ),
    TokenType.NOT_EQUAL:     ParseRule(None,      binary, Precedence.EQUALITY   ),

    TokenType.INTEGER:       ParseRule(integer,   None,   Precedence.NONE       ),
    TokenType.FLOAT:         ParseRule(float_,    None,   Precedence.NONE       ),
    TokenType.COMPLEX:       ParseRule(complex_,  None,   Precedence.NONE       ),
    TokenType.TRUE:          ParseRule(literal,   None,   Precedence.NONE       ),
    TokenType.FALSE:         ParseRule(literal,   None,   Precedence.NONE       ),

    TokenType.IF:            ParseRule(None,      None,   Precedence.NONE       ),
    TokenType.ELSE:          ParseRule(None,      None,   Precedence.NONE       ),
    TokenType.THEN:          ParseRule(None,      None,   Precedence.NONE       ),

    TokenType.ASSERT:        ParseRule(None,      None,   Precedence.NONE       ),
    TokenType.PRINT:         ParseRule(None,      None,   Precedence.NONE       ),
    TokenType.NEWLINE:       ParseRule(None,      None,   Precedence.NONE       ),
    TokenType.EOF:           ParseRule(None,      None,   Precedence.NONE       ),

}
# fmt: one
