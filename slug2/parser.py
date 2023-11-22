from enum import IntEnum, auto
from typing import TYPE_CHECKING, Callable, NewType, cast

from slug2.common import (
    CompilerError,
    ConstantIndex,
    FuncType,
    JumpDistance,
    LocalIndex,
    NumArgs,
    Op,
    ParseError,
    UpvalueIndex,
)
from slug2.compiler import Compiler
from slug2.token import Token, TokenType, tokenize

if TYPE_CHECKING:
    from slug2.vm import VM


class Precedence(IntEnum):
    NONE = auto()
    ASSIGNMENT = auto()  # =
    OR = auto()  # or
    AND = auto()  # and
    EQUALITY = auto()  # == !=
    COMPARISON = auto()  # > >= < <=
    TERM = auto()  # +, -
    FACTOR = auto()  # * / //
    EXPONENT = auto()  # **
    UNARY = auto()  # - not
    CALL = auto()  # . ()
    PRIMARY = auto()

    def next_precedence(self) -> "Precedence":
        value = self.value + 1
        for pt in Precedence:
            if pt.value == value:
                return pt
        raise ParseError("unknown precedence")


class Parser:
    __slots__ = ("vm", "tokens", "current_index", "had_error", "panic_mode")

    def __init__(self, vm: "VM", source: str | None = None) -> None:
        self.vm = vm
        self.tokens: list[Token] = [] if source is None else tokenize(source)
        self.current_index: int = -1
        self.had_error: bool = False
        self.panic_mode: bool = False

        if __debug__:
            print("TOKENS:")
            for t in self.tokens:
                print(f"  {t}")
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

    def argument_list(self) -> int:
        arg_count = 0
        if not self.check(TokenType.RIGHT_PAREN):
            self.expression()
            arg_count += 1
            while self.match(TokenType.COMMA):
                self.expression()
                if self.argument_list == 255:
                    raise ParseError("cannot have more than 255 arguments")
                arg_count += 1
        self.consume(TokenType.RIGHT_PAREN, "expect ) after arguments")
        return arg_count

    def strip_current_newlines(self) -> None:
        strip: int = 0
        while self.peek(strip).tokentype == TokenType.NEWLINE:
            strip += 1

        self.tokens = (  # stripping the current consecutive newlines from tokens
            self.tokens[: self.current_index] + self.tokens[self.current_index + strip :]
        )

    def parse_precedence(self, precedence: Precedence, group: bool = False) -> None:
        if __debug__:
            print(f"parse_precedence :{precedence} :{group} on line {self.peek(0).line}")

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
        self.vm.compiler.emit_byte(Op.POP)

    def assert_statement(self) -> None:
        self.expression()
        self.vm.compiler.emit_byte(Op.ASSERT)

    def print_statement(self):
        self.expression()
        self.vm.compiler.emit_byte(Op.PRINT)

    def skip_newlines(self) -> None:
        while self.peek().tokentype == TokenType.NEWLINE:
            self.current_index += 1

    def if_statement(self) -> None:
        jumps_to_end: list[JumpDistance] = []

        while True:  # in if-statement right after the if
            self.expression()
            self.consume(TokenType.NEWLINE, "expect newline after if clause")

            jump_to_next_clause = self.vm.compiler.emit_jump(Op.JUMP_IF_FALSE)
            self.vm.compiler.emit_byte(Op.POP)

            self.block([TokenType.ELSE, TokenType.END])

            jumps_to_end.append(self.vm.compiler.emit_jump(Op.JUMP))

            self.vm.compiler.patch_jump(jump_to_next_clause)
            self.vm.compiler.emit_byte(Op.POP)

            if self.match(TokenType.ELSE):  # consumes the else
                if self.match(TokenType.IF):  # consumes the if
                    continue  # else if statement -> back to top of loop
                else:  # bare else statement -> fall through to break
                    self.block([TokenType.END])

            break

        self.consume(TokenType.END, "expect end after if block")

        for jump in jumps_to_end:
            self.vm.compiler.patch_jump(jump)

    def return_statement(self) -> None:
        if self.vm.compiler.function.functype == FuncType.SCRIPT:
            raise ParseError("cannot return from top-level code")

        if self.match(TokenType.NEWLINE):
            self.vm.compiler.emit_return()
        else:
            if self.vm.compiler.function.functype == FuncType.INITIALIZER:
                raise ParseError("cannot return from an initializer")
            self.expression()
            self.consume(TokenType.NEWLINE, "expect newline after return")
            self.vm.compiler.emit_byte(Op.RETURN)

    def statement(self) -> None:
        if self.match(TokenType.NEWLINE):
            pass
        elif self.match(TokenType.ASSERT):
            self.assert_statement()
        elif self.match(TokenType.PRINT):
            self.print_statement()
        elif self.match(TokenType.IF):
            self.if_statement()
        elif self.match(TokenType.LEFT_BRACE):
            compiler = self.vm.compiler
            if compiler is None:
                raise CompilerError("compiler is None")
            compiler.begin_scope()
            self.block([TokenType.RIGHT_BRACE])
            self.consume(TokenType.RIGHT_BRACE, "no closing }")
            compiler.end_scope()
        elif self.match(TokenType.RETURN):
            self.return_statement()
        else:
            self.expression_statement()

    def parse_variable(self, error_message: str) -> ConstantIndex:
        if __debug__:
            print(f"parse_variable on line {self.peek(0).line}")

        self.consume(TokenType.IDENTIFIER, error_message)
        previous = self.peek(-1)

        self.vm.compiler.declare_variable(previous)
        if self.vm.compiler.scope_depth > 0:
            return ConstantIndex(0)

        return self.vm.compiler.identifier_constant(previous)

    def variable_declaration(self) -> None:
        if __debug__:
            print(f"variable_declaration on line {self.peek(0).line}")

        constant_index = self.parse_variable("expect variable name")
        self.consume(TokenType.EQUAL, "expect = after variable name")
        self.expression()

        self.vm.compiler.define_variable(constant_index)

        if __debug__:
            print("\nLOCALS:")
            for idx, local in enumerate(self.vm.compiler.locals):
                if local is not None:
                    print(f"  {idx} -> {local}")
            print()

    def function(self, functype: FuncType) -> None:
        if __debug__:
            print(f"function :name {self.peek(-1).literal} :type {functype}")

        self.vm.compiler = Compiler(self.vm, self.peek(-1).literal, functype)
        self.vm.compiler.begin_scope()

        while not self.match(TokenType.EQUAL):
            self.vm.compiler.function.arity += 1
            if self.vm.compiler.function.arity > 255:
                raise ParseError("can't have more than 255 parameters")
            constant_index = self.parse_variable("expect parameter name")
            previous = self.peek(-1)
            print("SAVING", previous.literal, "at", constant_index, "with depth", self.vm.compiler.scope_depth)
            self.vm.compiler.define_variable(constant_index)

        self.consume(TokenType.LEFT_BRACE, "expect { after = in function declaration")
        self.block([TokenType.RIGHT_BRACE])
        self.consume(TokenType.RIGHT_BRACE, "expect } after function block")

        function = self.vm.compiler.end()
        self.vm.compiler.emit_bytes(Op.CLOSURE, self.vm.compiler.make_constant(function))

        for idx in range(function.num_upvalues):
            upvalue = self.vm.compiler.upvalues[idx]
            if upvalue is None:
                print(function)
                print("EMITTING UPALUE:", upvalue, idx)
                print(self.vm.compiler.upvalues[:5])

                raise RuntimeError("unexpectedly None upvalue")

            self.vm.compiler.emit_upvalue(upvalue.is_local, upvalue.index)

    def function_declaration(self) -> None:
        if __debug__:
            print(f"function_declaration on line {self.peek(0).line}")

        constant_index = self.parse_variable("expect function name")
        self.vm.compiler.mark_initialized()
        self.function(FuncType.FUNCTION)
        self.vm.compiler.define_variable(constant_index)

    def declaration(self) -> None:
        if self.match(TokenType.LET):
            self.variable_declaration()
        if self.match(TokenType.FN):
            self.function_declaration()
        else:
            self.statement()

    def block(self, end_tokentypes: list[TokenType]) -> None:
        if __debug__:
            print(f"block on line {self.peek(0).line}")

        end_tokentypes_or_eof = set(end_tokentypes + [TokenType.EOF])
        while not self.check_multiple(end_tokentypes_or_eof):
            self.declaration()

    def named_variable(self, name: Token, can_assign: bool) -> None:
        if __debug__:
            print(f"named_variable :{name.literal} on line {self.peek(0).line}")

        arg: LocalIndex | ConstantIndex | UpvalueIndex | None
        if (arg := self.vm.compiler.resolve_local(name)) is not None:
            get_op = Op.GET_LOCAL
            set_op = Op.SET_LOCAL
        elif (arg := self.vm.compiler.resolve_upvalue(name)) is not None:
            get_op = Op.GET_UPVALUE
            set_op = Op.SET_UPVALUE
        else:
            arg = self.vm.compiler.identifier_constant(name)
            get_op = Op.GET_GLOBAL
            set_op = Op.SET_GLOBAL

        if can_assign and self.match(TokenType.EQUAL):
            self.expression()
            self.vm.compiler.emit_bytes(set_op, arg)
        else:
            self.vm.compiler.emit_bytes(get_op, arg)


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
            parser.vm.compiler.emit_byte(Op.ADD)
        case TokenType.MINUS:
            parser.vm.compiler.emit_byte(Op.SUBTRACT)
        case TokenType.STAR:
            parser.vm.compiler.emit_byte(Op.MULTIPLY)
        case TokenType.STAR_STAR:
            parser.vm.compiler.emit_byte(Op.EXPONENT)
        case TokenType.SLASH:
            parser.vm.compiler.emit_byte(Op.DIVIDE)
        case TokenType.SLASH_SLASH:
            parser.vm.compiler.emit_byte(Op.INT_DIVIDE)
        case TokenType.LESS:
            parser.vm.compiler.emit_byte(Op.LESS)
        case TokenType.LESS_EQUAL:
            parser.vm.compiler.emit_byte(Op.LESS_EQUAL)
        case TokenType.GREATER:
            parser.vm.compiler.emit_byte(Op.GREATER)
        case TokenType.GREATER_EQUAL:
            parser.vm.compiler.emit_byte(Op.GREATER_EQUAL)
        case TokenType.EQUAL_EQUAL:
            parser.vm.compiler.emit_byte(Op.VALUE_EQUAL)
        case TokenType.NOT_EQUAL:
            parser.vm.compiler.emit_byte(Op.NOT_VALUE_EQUAL)
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
        case TokenType.MINUS:
            parser.vm.compiler.emit_byte(Op.NEGATE)
        case _:
            raise ValueError("unknown unary operator")


def literal(parser: Parser, _: bool) -> None:
    previous = parser.peek(-1)

    if __debug__:
        print(f"literal: {previous.value} on line {previous.line}")

    match tokentype := previous.tokentype:
        case TokenType.TRUE:
            parser.vm.compiler.emit_byte(Op.TRUE)
        case TokenType.FALSE:
            parser.vm.compiler.emit_byte(Op.FALSE)
        case _:
            RuntimeError(f"unknown literal {tokentype}")


def integer(parser: Parser, _: bool) -> None:
    previous = parser.peek(-1)

    if __debug__:
        print(f"integer: {previous} on line {previous.line}")

    parser.vm.compiler.emit_constant(previous.value)


def _float(parser: Parser, _: bool) -> None:
    previous = parser.peek(-1)

    if __debug__:
        print(f"float: {previous} on line {previous.line}")

    parser.vm.compiler.emit_constant(previous.value)


def _complex(parser: Parser, _: bool) -> None:
    previous = parser.peek(-1)

    if __debug__:
        print(f"complex: {previous} on line {previous.line}")

    parser.vm.compiler.emit_constant(previous.value)


def grouping(parser: Parser, _: bool) -> None:
    parser.expression(group=True)
    parser.consume(TokenType.RIGHT_PAREN, "didn't find closing )")


def variable(parser: Parser, can_assign: bool) -> None:
    previous = parser.peek(-1)

    if __debug__:
        print(f"variable: {previous.literal} on line {previous.line}")

    parser.named_variable(previous, can_assign)


def call(parser: Parser, _: bool) -> None:
    arg_count = parser.argument_list()
    parser.vm.compiler.emit_bytes(Op.CALL, NumArgs(arg_count))


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
    TokenType.LEFT_BRACE:    ParseRule(None,      None,   Precedence.NONE       ),
    TokenType.RIGHT_BRACE:   ParseRule(None,      None,   Precedence.NONE       ),

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
    TokenType.FLOAT:         ParseRule(_float,    None,   Precedence.NONE       ),
    TokenType.COMPLEX:       ParseRule(_complex,  None,   Precedence.NONE       ),
    TokenType.TRUE:          ParseRule(literal,   None,   Precedence.NONE       ),
    TokenType.FALSE:         ParseRule(literal,   None,   Precedence.NONE       ),
    TokenType.IDENTIFIER:    ParseRule(variable,  None,   Precedence.NONE       ),

    TokenType.IF:            ParseRule(None,      None,   Precedence.NONE       ),
    TokenType.ELSE:          ParseRule(None,      None,   Precedence.NONE       ),

    TokenType.ASSERT:        ParseRule(None,      None,   Precedence.NONE       ),
    TokenType.PRINT:         ParseRule(None,      None,   Precedence.NONE       ),
    TokenType.LET:           ParseRule(None,      None,   Precedence.NONE       ),

    TokenType.NEWLINE:       ParseRule(None,      None,   Precedence.NONE       ),
    TokenType.COMMA:         ParseRule(None,      None,   Precedence.NONE       ),

    TokenType.EOF:           ParseRule(None,      None,   Precedence.NONE       ),

}
# fmt: one
