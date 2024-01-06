from pytest import raises

from slug2.vm import VM


def test_associativity():
    vm = VM()
    vm.interpret(
        """
let a = 1
let b = 2
let c = 3

a = b = c

assert a == 3
assert b == 3
assert c == 3
    """.strip()
    )


def test_global():
    vm = VM()
    vm.interpret(
        """
let a = 1
assert a == 1

a = 2
assert a === 2

print a = 3
assert a == 3
    """
    )


def test_undefined():
    vm = VM()
    with raises(RuntimeError):
        vm.interpret("""unknown = 1""")
