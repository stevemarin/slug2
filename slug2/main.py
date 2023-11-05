from slug2 import vm


def test_assert() -> None:
    vm.interpret("""assert 1 == 1""")


def test_assert_arithmetic() -> None:
    vm.interpret("""assert 1 + 3 * 2 / 6 - 12 / 3 - 1 == -3 / 5 * 5""")


def test_double_negative() -> None:
    vm.interpret("""assert 1 == --1""")


def test_failed_assert() -> None:
    from pytest import raises

    with raises(AssertionError):
        vm.interpret("""assert 1 == 0""")


if __name__ == "__main__":
    # vm.interpret("""assert (3 + 2 + 1) / -2 == -3.0""")

    vm.interpret("""(1 + 1j) * (1 + -1j)""")
