from slug2 import vm


def test_precedence():
    vm.interpret("""assert -1**2 == 1""")
    vm.interpret("""assert -(1**2) == -1""")
    vm.interpret("""assert 1 - -1 == 2""")
    vm.interpret("""assert 1 + 2 * -3 == -5""")
    vm.interpret("""assert 2 ** (-1 + 3) + 1 == 5""")

def test_multiline():
    vm.interpret("""
        assert 1 == 1
        assert 2 == 2
    """)

    vm.interpret("""
        assert 1 == 1 + 1 ---2 + 1 
        assert 2 == 2
        1 == 1
        assert 1  +  1
                  +  3
                  == 5
    """)

def test_if():
    from slug2 import vm
    vm.interpret("""
        if 1 == 1
            print True
        else
            print False
        print 12345
    """)