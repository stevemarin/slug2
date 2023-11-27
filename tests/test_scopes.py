from slug2.vm import VM

vm = VM()


def test_scopes():
    vm.interpret(
        """
        let x = 0
        assert x == 0
        {
            assert x == 0
            let x = 1
            assert x == 1
            {
                assert x == 1
                let x = 2
                assert x == 2
            }
            assert x == 1
        }
        assert x == 0
        """.strip()
    )
