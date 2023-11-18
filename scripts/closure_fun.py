from collections import namedtuple
from typing import Any

import pyperf


class _Cat:
    pass


def Cat(_name: str):
    def name_getter(_name: str):
        def name():
            return _name

        return name

    name = name_getter(_name)

    def speak() -> str:
        return f"'Meow...' said {_name}"

    attrs: dict[str, Any] = {"name": name, "speak": speak}

    Cat = namedtuple("Cat", list(attrs.keys()))  # type: ignore
    cat = Cat(**attrs)

    # cat = _Cat()
    # setattr(cat, "name", _name)
    # setattr(cat, "speak", speak)

    return cat


class Cat2:
    __slots__ = "name"

    def __init__(self, name: str):
        self.name = name

    def speak(self) -> str:
        return f"'Meow...' said {self.name}"


runner = pyperf.Runner()


def c_runner():
    c = Cat("Fluffy")
    c.speak()
    c.name()


runner.bench_func("closure", c_runner)


def c_runner2():
    c2 = Cat2("Fluffy2")
    c2.speak()
    c2.name


runner.bench_func("class", c_runner2)
