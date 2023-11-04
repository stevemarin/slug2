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
    FOR = "for"
    WHILE = "while"
    IF = "if"
    THEN = "then"
    ELIF = "elif"
    ELSE = "else"
    END = "end"
    DEFER = "defer"
    FINALLY = "finally"
    WITH = "with"

    # types
    INTEGER = "integer"
    FLOAT = "float"
    COMPLEX = "complex"

    # type conversion
    AS = "as"
    UNION = "union"
    INTERSECTION = "intersection"

    # statements
    ASSERT = "assert"

    # other
    IDENTIFIER = "identifier"
    EOF = "eof"

    @staticmethod
    def get(token: str):
        for tt in TokenType:
            if tt.value == token:
                return tt
        raise TokenizationError("token {token} not found in TokenType")


KWT = KeywordTree("")
for token in TokenType:
    KWT.add_token(token.value)

DIGITS = frozenset(digits)
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
        return f"<Token :{self.tokentype} :'{self.literal}'>"


def tokenize(source: str) -> list[Token]:
    current_idx: int = 0
    line: int = 0
    source_length: int = len(source)

    tokens: list[Token] = []

    def at_end(offset: int = 0) -> bool:
        return False if current_idx + offset < source_length else True

    def get_number() -> tuple[int, Token]:
        offset = 1
        while not at_end(offset) and source[current_idx + offset] in DIGITS:
            offset += 1

        literal = source[current_idx : current_idx + offset]
        token = Token(TokenType.INTEGER, literal, int(literal), current_idx, line)

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


def test_tokenize_plus_minus():
    tokens = tokenize("""11 + 3 - 7""")

    assert len(tokens) == 6

    assert tokens[0].tokentype == TokenType.INTEGER
    assert tokens[0].literal == "11"
    assert tokens[0].value == 11

    assert tokens[1].tokentype == TokenType.PLUS
    assert tokens[1].literal == "+"
    assert tokens[1].value is None

    assert tokens[2].tokentype == TokenType.INTEGER
    assert tokens[2].literal == "3"
    assert tokens[2].value == 3

    assert tokens[3].tokentype == TokenType.MINUS
    assert tokens[3].literal == "-"
    assert tokens[3].value is None

    assert tokens[4].tokentype == TokenType.INTEGER
    assert tokens[4].literal == "7"
    assert tokens[4].value == 7

    assert tokens[5].tokentype == TokenType.EOF


def test_tokenize_unexpected():
    from pytest import raises

    with raises(TokenizationError):
        tokenize("""1 + 3 !""")


def test_more_types():
    tokens = tokenize("""3==!=.....3;^""")
    expected_types = [
        TokenType.INTEGER,
        TokenType.EQUAL_EQUAL,
        TokenType.NOT_EQUAL,
        TokenType.ELLIPSIS,
        TokenType.DOT_DOT,
        TokenType.INTEGER,
        TokenType.SEMICOLON,
        TokenType.CARET,
        TokenType.EOF,
    ]
    for token, expected_type in zip(tokens, expected_types):
        assert token.tokentype == expected_type


if __name__ == "__main__":
    for t in tokenize("""3==!=.....3;^"""):
        print(t)
