from slug2.vm import VM

vm = VM()


def test_close_over_global():
    vm.interpret(
        """
    let x = 5
    fn five = {
        return x
    }

    assert five() == 5

    x = 6
    assert five() == 6
    # """.strip()
    )
