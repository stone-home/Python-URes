import pytest
from typing import (
    Callable,
    Any,
    Type,
    Dict,
    Union,
    get_type_hints,
    Optional,
    List,
    Tuple,
)
from ures.tools.decorator import type_check, check_instance_variable


# ============================ Decorator for Instance Variable Checking ===========================
class TestDecoratorWithExistingVariable:
    def test_variable_is_not_none(self):
        class TestClass:
            def __init__(self):
                self.data = "some data"

            @check_instance_variable("data")
            def process_data(self):
                return f"Processing: {self.data}"

        instance = TestClass()
        assert instance.process_data() == "Processing: some data"

    def test_variable_is_none(self):
        class TestClass:
            def __init__(self):
                self.data = None

            @check_instance_variable("data")
            def process_data(self):
                return f"Processing: {self.data}"

        instance = TestClass()
        assert instance.process_data() is None


class TestDecoratorWithMultipleDecorators:
    def test_all_variables_exist_and_are_not_none(self):
        class TestClass:
            def __init__(self):
                self.value1 = 10
                self.value2 = "hello"

            @check_instance_variable("value1")
            @check_instance_variable("value2")
            def combine_values(self):
                return f"{self.value1} - {self.value2}"

        instance = TestClass()
        assert instance.combine_values() == "10 - hello"

    def test_one_variable_is_none(self):
        class TestClass:
            def __init__(self):
                self.value1 = None
                self.value2 = "hello"

            @check_instance_variable("value1")
            @check_instance_variable("value2")
            def combine_values(self):
                return f"{self.value1} - {self.value2}"

        instance = TestClass()
        assert instance.combine_values() is None


class TestDecoratorWithMethodArguments:
    def test_with_positional_argument_variable_not_none(self):
        class TestClass:
            def __init__(self):
                self.data = "initial data"

            @check_instance_variable("data")
            def process_with_arg(self, prefix):
                return f"{prefix}: {self.data}"

        instance = TestClass()
        assert instance.process_with_arg("Output") == "Output: initial data"

    def test_with_positional_argument_variable_is_none(self):
        class TestClass:
            def __init__(self):
                self.data = None

            @check_instance_variable("data")
            def process_with_arg(self, prefix):
                return f"{prefix}: {self.data}"

        instance = TestClass()
        assert instance.process_with_arg("Output") is None

    def test_with_keyword_argument_variable_not_none(self):
        class TestClass:
            def __init__(self):
                self.config = {"setting": True}

            @check_instance_variable("config")
            def get_setting(self, key):
                return self.config.get(key)

        instance = TestClass()
        assert instance.get_setting(key="setting") is True

    def test_with_keyword_argument_variable_is_none(self):
        class TestClass:
            def __init__(self):
                self.config = None

            @check_instance_variable("config")
            def get_setting(self, key):
                return self.config.get(key)

        instance = TestClass()
        assert instance.get_setting(key="setting") is None


# =========================== Decorator for Type Checking ===========================
class TestCorrectUsage:
    def test_correct_usage(self):
        @type_check()
        def my_function(name: str, age: int, height: Union[int, float]) -> None:
            pass

        my_function("Alice", 30, 165.5)

    def test_correct_usage_list(self):
        @type_check()
        def my_function(data: List[int]) -> None:
            pass

        my_function([1, 2, 3])

    def test_correct_usage_tuple(self):
        @type_check()
        def my_function(point: Tuple[int, int]) -> None:
            pass

        my_function((1, 2))

    def test_correct_usage_union(self):
        @type_check()
        def my_function(value: Union[int, float]) -> None:
            pass

        my_function(1)
        my_function(1.2)


class TestIncorrectUsage:
    def test_incorrect_usage_basic(self):
        @type_check()
        def my_function(name: str, age: int) -> None:
            pass

        with pytest.raises(TypeError):
            my_function("Alice", "30")

    def test_incorrect_usage_union(self):
        @type_check()
        def my_function(value: Union[int, float]) -> None:
            pass

        with pytest.raises(TypeError):
            my_function("abc")

    def test_incorrect_usage_list(self):
        @type_check()
        def my_function(data: List[int]) -> None:
            pass

        with pytest.raises(TypeError):
            my_function([1, 2, "3"])

    def test_incorrect_usage_tuple(self):
        @type_check()
        def my_function(point: Tuple[int, int]) -> None:
            pass

        with pytest.raises(TypeError):
            my_function((1, "2"))


class TestSkipArgs:
    def test_skip_args(self):
        @type_check(skip_args=["age"])
        def my_function(name: str, age: int) -> None:
            pass

        my_function("Alice", "30")
        with pytest.raises(TypeError):
            my_function(123, 30)

    def test_skip_multiple_args(self):
        @type_check(skip_args=["age", "name"])
        def my_function(name: str, age: int, value: float) -> None:
            pass

        my_function("Alice", "30", 1.2)
        with pytest.raises(TypeError):
            my_function("Alice", "30", "abc")


class TestWrongInput:
    def test_wrong_skip_args(self):
        with pytest.raises(ValueError):

            @type_check(skip_args=[123])
            def my_function(value: Union[int, float]) -> None:
                pass
