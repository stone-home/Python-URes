import pytest
from enum import Enum
from ures.tools.enum import EnumManipulator


# Define a sample Enum for testing.
class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


@pytest.fixture
def manipulator():
    return EnumManipulator(Color)


def test_fetch_keys(manipulator):
    keys = manipulator.fetch_keys()
    assert isinstance(keys, list)
    assert set(keys) == {"RED", "GREEN", "BLUE"}


def test_fetch_enum_case_insensitive(manipulator):
    member = manipulator.fetch_enum("red")
    assert member is not None
    assert member.value == 1
    # Test with different case.
    member = manipulator.fetch_enum("GrEeN")
    assert member is not None
    assert member.value == 2


def test_check_key(manipulator):
    assert manipulator.check_key("BLUE") is True
    assert manipulator.check_key("PURPLE") is False


def test_fetch_value(manipulator):
    value = manipulator.fetch_value("GREEN")
    assert value == 2
    value_none = manipulator.fetch_value("YELLOW")
    assert value_none is None


def test_filter_by_without_field(manipulator):
    # In this case, string representation of values ("1", "2", "3") is used.
    # Since keyword "1" does not match exactly (because "1" != "1" as a string?),
    # we'll add a custom Enum where values are strings.
    class Fruit(Enum):
        APPLE = "red"
        BANANA = "yellow"
        GRAPE = "purple"

    fruit_manipulator = EnumManipulator(Fruit)
    result = fruit_manipulator.filter_by("red")
    assert result == ["APPLE"]


def test_filter_by_with_field():
    # For testing 'field', we'll create an Enum with complex values.
    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age

    class People(Enum):
        ALICE = Person("Alice", 30)
        BOB = Person("Bob", 25)
        CAROL = Person("Carol", 40)

    person_manipulator = EnumManipulator(People)
    # Filter by name attribute.
    result = person_manipulator.filter_by("bob", field="name")
    assert result == ["BOB"]
    # Filtering with a keyword not matching any.
    result = person_manipulator.filter_by("David", field="name")
    assert result == []
