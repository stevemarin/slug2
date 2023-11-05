from pytest import raises

from slug2 import vm


def test_assert() -> None:
    vm.interpret("""assert 1 == 1""")


def test_assert_arithmetic() -> None:
    vm.interpret("""assert 1 + 3 * 2 / 6 - 12 / 3 - 1 == -3 / 5 * 5""")


def test_double_negative() -> None:
    vm.interpret("""assert 1 == --1""")


def test_failed_assert() -> None:
    with raises(AssertionError):
        vm.interpret("""assert 1 == 0""")
