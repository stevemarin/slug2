
from slug2 import vm

def test_precedence():
    vm.interpret("""assert -1**2 == 1""")
    vm.interpret("""assert -(1**2) == -1""")
    vm.interpret("""assert 1 - -1 == 2""")
    vm.interpret("""assert 1 + 2 * -3 == -5""")
    vm.interpret("""assert 2 ** (-1 + 3) + 1 == 5""")
    