
from slug2.token import TokenType, tokenize, TokenizationError

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

