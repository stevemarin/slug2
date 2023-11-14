from enum import Enum
from string import ascii_letters, digits
from typing import Any


class TokenizationError(Exception):
    pass


class KeywordTree:
    __slots__ = ("char", "token", "children")

    def __init__(self, char: str):
        self.char = char
        self.token: str | None = None
        self.children: dict["str", "KeywordTree"] = {}

    def add_token(self, token: str) -> None:
        current = self
        for char in token:
            if char in current.children:
                current = current.children[char]
            else:
                current.children[char] = KeywordTree(char)
                current = current.children[char]

        current.token = token

    def __getitem__(self, token: str) -> "KeywordTree":
        current = self
        for idx, char in enumerate(token):
            try:
                current = current.children[char]
            except KeyError:
                raise KeyError(f"error at index {idx}, char {char} not found")

        return current


class TokenType(Enum):
    # parens
    LEFT_PAREN = "("
    RIGHT_PAREN = ")"
    LEFT_SQUARE = "["
    RIGHT_SQUARE = "]"
    LEFT_BRACE = "{"
    RIGHT_BRACE = "}"

    # quotes
    SINGLE_QUOTE = "'"
    DOUBLE_QUOTE = '"'
    TRIPLE_DOUBLE_QUOTE = '"""'
    TRIPLE_SINGLE_QUOTE = "'''"

    # comments
    POUND = "#"

    # symbols
    PLUS = "+"
    PLUS_EQUAL = "+="
    MINUS = "-"
    ARROW_RIGHT = "->"
    ARROW_LEFT = "<-"
    MINUS_EQUAL = "-="
    STAR = "*"
    STAR_EQUAL = "*="
    STAR_STAR = "**"
    SLASH = "/"
    SLASH_EQUAL = "/="
    SLASH_SLASH = "//"
    EQUAL = "="
    EQUAL_EQUAL = "=="
    NOT_EQUAL = "!="
    LESS = "<"
    LESS_LESS = "<<"
    LESS_EQUAL = "<="
    GREATER = ">"
    GREATER_GREATER = ">>"
    GREATER_EQUAL = ">="
    AT = "@"
    CARET = "^"
    DOLLAR = "$"
    DOT = "."
    DOT_DOT = ".."
    ELLIPSIS = "..."
    COLON = ":"
    COLON_COLON = "::"
    SEMICOLON = ";"
    COMMA = ","
    AMPERSAND = "&"
    AMPERSAND_AMPERSAND = "&&"
    PIPE = "|"
    PIPE_PIPE = "||"

    # whitespace
    SPACE = " "
    TAB = "\t"
    NEWLINE = "\n"

    # logicals
    NOT = "not"
    AND = "and"
    OR = "or"

    # control flow
    IF = "if"
    THEN = "then"
    ELIF = "elif"
    ELSE = "else"
    FOR = "for"
    WHILE = "while"
    END = "end"
    DEFER = "defer"
    FINALLY = "finally"
    WITH = "with"

    # types
    INTEGER = "integer"
    FLOAT = "float"
    COMPLEX = "complex"
    TRUE = "True"
    FALSE = "False"

    # type conversion
    AS = "as"
    UNION = "union"
    INTERSECTION = "intersection"

    # statements
    ASSERT = "assert"
    PRINT = "print"

    # other
    IDENTIFIER = "identifier"
    EOF = "eof"

    @staticmethod
    def get(token: str):
        for tt in TokenType:
            if tt.value == token:
                return tt
        raise TokenizationError(f"token {token} not found in TokenType")


KWT = KeywordTree("")
for token in TokenType:
    KWT.add_token(token.value)

DIGITS = frozenset(digits)
DIGITS_UNDERSCORE = frozenset(DIGITS.union("_"))
LETTERS = frozenset(ascii_letters)
FIRST_CHARS = frozenset(LETTERS.union("_"))
ALPHANUM = frozenset(FIRST_CHARS.union(DIGITS))


class Token:
    __slots__ = ("tokentype", "literal", "value", "start", "line")

    def __init__(self, tokentype: TokenType, literal: str, value: Any, start: int, line: int):
        self.tokentype = tokentype
        self.literal = literal
        self.value = value
        self.start = start
        self.line = line

    def __len__(self) -> int:
        return len(self.literal)

    def __repr__(self) -> str:
        return f"<Token :{self.tokentype}>"


def tokenize(source: str) -> list[Token]:
    current_idx: int = 0
    line: int = 0
    source_length: int = len(source)

    tokens: list[Token] = []

    def newline() -> Token:
        return Token(TokenType.NEWLINE, "\n", None, current_idx, line)

    def at_end(offset: int = 0) -> bool:
        return False if current_idx + offset < source_length else True

    def get_number() -> tuple[int, Token]:
        found_dot = False
        found_j = False

        # we already know offset = 0 is a DIGIT
        offset = 1

        # integer part
        for char in source[current_idx + offset :]:
            if char not in DIGITS_UNDERSCORE:
                break
            else:
                offset += 1

        # decimal part
        if not at_end(offset) and source[current_idx + offset] == ".":
            found_dot = True
            offset += 1
            for char in source[current_idx + offset :]:
                if char not in DIGITS_UNDERSCORE:
                    break
                else:
                    offset += 1

        # scientific notation part
        if not at_end(offset) and source[current_idx + offset] in frozenset("eE"):
            offset += 1
            if source[current_idx + offset] in frozenset("-+"):
                offset += 1
            for char in source[current_idx + offset :]:
                if char not in DIGITS_UNDERSCORE:
                    break
                else:
                    offset += 1

        # complex part
        if not at_end(offset) and source[current_idx + offset] in frozenset("jJ"):
            found_j = True
            offset += 1

        token_str = source[current_idx : current_idx + offset]

        if found_j:
            tokentype = TokenType.COMPLEX
            value = complex(token_str)
        elif found_dot:
            tokentype = TokenType.FLOAT
            value = float(token_str)
        else:
            tokentype = TokenType.INTEGER
            value = int(token_str)

        token = Token(tokentype, token_str, value, current_idx, line)
        return offset, token

    def until_not_chars(chars: set[str] | frozenset[str]) -> str:
        for length, c in enumerate(source[current_idx:]):
            if c not in chars:
                return source[current_idx : current_idx + length]
        return source[current_idx:]

    def remaining_chars() -> str:
        return source[current_idx:]

    def get_reserved_token() -> None | Token:
        kwt_node = KWT
        token_str: str | None = None

        num_remaining_chars = len(remaining_chars())
        if num_remaining_chars == 1:
            kwt_node = KWT[source[current_idx]]
            if kwt_node.token is not None:
                token_str = kwt_node.token
        else:
            for distance in range(num_remaining_chars):
                try:
                    current_token_str = source[current_idx : current_idx + distance]
                    kwt_node = KWT[current_token_str]
                    if kwt_node.token is not None:
                        token_str = kwt_node.token
                except KeyError:
                    break

        return None if token_str is None else Token(TokenType.get(token_str), token_str, None, current_idx, line)

    while not at_end():
        match char := source[current_idx]:
            case " " | "\t":
                current_idx += 1
            case "\n":
                tokens.append(newline())
                current_idx += 1
                line += 1
            case char if char in DIGITS:
                offset, token = get_number()
                tokens.append(token)
                current_idx += offset
            case char if char in FIRST_CHARS:
                token_str = until_not_chars(ALPHANUM)
                tokentype = TokenType.get(token_str)
                token = Token(tokentype, char, None, current_idx, line)
                tokens.append(token)
                current_idx += len(token_str)
            case _:
                maybe_token = get_reserved_token()
                if maybe_token is None:
                    start = max(0, current_idx - 5)
                    end = min(len(source), current_idx + 5)
                    context = source[start:end]
                    raise TokenizationError(f"unexpected token at {current_idx}: {context}")
                tokens.append(maybe_token)
                current_idx += len(maybe_token.literal)

    tokens.append(Token(TokenType.EOF, "", None, current_idx, line))

    return tokens
