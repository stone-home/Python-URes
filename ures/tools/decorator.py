import logging
import inspect
from functools import wraps
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


logger = logging.getLogger(__file__)


def check_instance_variable(variable_name):
    """
    A decorator to check if an instance variable is None before executing a method.

    Args:
        variable_name: The name of the instance variable (as a string) to check.

    Returns:
        The decorated method if the instance variable is not None, or None if it is.
        Alternatively, you can raise an exception.
    """

    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            try:
                # Use hasattr to check if the instance has the attribute, and then getattr to get it
                if (
                    hasattr(self, variable_name)
                    and getattr(self, variable_name) is None
                ):
                    # Option 1: Return None
                    return None

                    # Option 2: Raise an exception (recommended)
                    # raise ValueError(f"Instance variable '{variable_name}' cannot be None.")

                elif not hasattr(
                    self, variable_name
                ):  # Handle the case where the attribute doesn't exist at all
                    raise AttributeError(
                        f"Instance variable '{variable_name}' does not exist."
                    )

            except (ValueError, AttributeError) as e:  # Catch both exceptions
                logger.error(f"Error in check_instance_variable decorator: {e}")
                return None  # Or re-raise the exception: raise

            return method(self, *args, **kwargs)

        return wrapper

    return decorator


def type_check(arg=None, skip_args: Optional[list[str]] = None) -> Callable:
    """
    Decorator to enforce data types of function arguments, extracting types from the function's annotation.

    Args:
        skip_args: An optional list of argument names to skip type checking for.
            Defaults to None (no arguments skipped).

    Returns:
        A decorator that wraps the function with type checking.

    Raises:
        TypeError:  If an argument's type does not match the expected type
                    (as specified in the function's type hints).
        ValueError: If a type in skip_args is not a string.
    """
    if callable(arg):
        # Decorator called directly without arguments (@type_check)
        func_to_wrap = arg
        skip_args_internal: list[str] = []
    else:
        # Decorator called with arguments (@type_check(skip_args=['...']))
        func_to_wrap = None
        skip_args_internal = skip_args if skip_args is not None else []

    if not all(isinstance(sa, str) for sa in skip_args_internal):
        raise ValueError("All arguments in skip_args must be strings.")

    keywords_to_skip = ["self"]  # List for future keywords to skip
    skip_check = set(skip_args_internal + keywords_to_skip)

    def decorator(func: Callable) -> Callable:
        """
        The actual decorator that wraps the function.
        """

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """
            The wrapper function that performs the type checking.
            """
            arg_dict: Dict[str, Any] = kwargs.copy()
            parameters = func.__code__.co_varnames[: func.__code__.co_argcount]
            arg_dict.update(dict(zip(parameters, args)))

            type_hints = get_type_hints(func)

            for arg_name, arg_value in arg_dict.items():
                if arg_name in type_hints and arg_name not in skip_check:
                    expected_type = type_hints[arg_name]
                    if isinstance(expected_type, type):  # check for base type
                        if not isinstance(arg_value, expected_type):
                            raise TypeError(
                                f"Argument '{arg_name}' should be of type {expected_type.__name__}, but got {type(arg_value).__name__}"
                            )
                    elif hasattr(
                        expected_type, "__origin__"
                    ):  #  Handles types like Union, List, Tuple
                        origin = expected_type.__origin__
                        if origin is Union:
                            types = expected_type.__args__
                            if not isinstance(arg_value, tuple(types)):
                                expected_types_str = ", ".join(
                                    t.__name__ for t in types
                                )
                                raise TypeError(
                                    f"Argument '{arg_name}' should be of type {expected_types_str}, "
                                    f"but got {type(arg_value).__name__}"
                                )
                        elif origin in (List, list):
                            if not isinstance(arg_value, list):
                                raise TypeError(
                                    f"Argument '{arg_name}' should be of type list, but got {type(arg_value).__name__}"
                                )
                            if expected_type.__args__ and not all(
                                isinstance(item, expected_type.__args__[0])
                                for item in arg_value
                            ):
                                raise TypeError(
                                    f"Argument '{arg_name}' should be a list of {expected_type.__args__[0].__name__}"
                                )
                        elif origin in (Tuple, tuple):
                            if not isinstance(arg_value, tuple):
                                raise TypeError(
                                    f"Argument '{arg_name}' should be of type tuple, but got {type(arg_value).__name__}"
                                )
                            if expected_type.__args__ and len(
                                expected_type.__args__
                            ) != len(arg_value):
                                raise TypeError(
                                    f"Argument '{arg_name}' should be a tuple of length {len(expected_type.__args__)}"
                                )
                            elif expected_type.__args__:
                                for i, expected_arg_type in enumerate(
                                    expected_type.__args__
                                ):
                                    if not isinstance(arg_value[i], expected_arg_type):
                                        raise TypeError(
                                            f"Argument '{arg_name}', element {i} should be of type {expected_arg_type}, but got {type(arg_value[i]).__name__}"
                                        )
                    else:
                        raise ValueError(
                            f"Type annotation for '{arg_name}' is not supported"
                        )
            return func(*args, **kwargs)

        return wrapper

    if func_to_wrap:
        return decorator(func_to_wrap)
    else:
        return decorator


# def type_check(skip_args: Optional[list[str]] = None) -> Callable:
#     """
#     Decorator to enforce data types of function arguments, extracting types from the function's annotation.
#
#     Args:
#         skip_args: An optional list of argument names to skip type checking for.
#             Defaults to None (no arguments skipped).
#
#     Returns:
#         A decorator that wraps the function with type checking.
#
#     Raises:
#         TypeError:  If an argument's type does not match the expected type
#                     (as specified in the function's type hints).
#         ValueError: If a type in skip_args is not a string.
#     """
#     if skip_args is None:
#         skip_args = []
#
#     keywords_to_skip = ["self"]  # List for future keywords to skip
#     skip_check = set(skip_args + keywords_to_skip)
#
#     if not all(isinstance(arg, str) for arg in skip_args):
#         raise ValueError("All arguments in skip_args must be strings.")
#
#     def decorator(func_to_wrap: Callable) -> Callable:
#         """
#         The actual decorator that wraps the function.
#         """
#         @wraps(func_to_wrap)
#         def wrapper(*args: Any, **kwargs: Any) -> Any:
#             """
#             The wrapper function that performs the type checking.
#             """
#             arg_dict: Dict[str, Any] = kwargs.copy()
#             parameters = func_to_wrap.__code__.co_varnames[:func_to_wrap.__code__.co_argcount]
#             arg_dict.update(dict(zip(parameters, args)))
#
#             type_hints = get_type_hints(func_to_wrap)
#
#             for arg_name, arg_value in arg_dict.items():
#                 if arg_name in type_hints and arg_name not in skip_args:
#                     if arg_name in ["self"]:
#                         # Skip type checking for keywords like 'self'
#                         continue
#                     expected_type = type_hints[arg_name]
#                     if isinstance(expected_type, type):  #check for base type
#                         if not isinstance(arg_value, expected_type):
#                             raise TypeError(f"Argument '{arg_name}' should be of type {expected_type.__name__}, but got {type(arg_value).__name__}")
#                     elif hasattr(expected_type, '__origin__'): #  Handles types like Union, List, Tuple
#                         origin = expected_type.__origin__
#                         if origin is Union:
#                             types = expected_type.__args__
#                             if not isinstance(arg_value, tuple(types)):
#                                 expected_types_str = ", ".join(t.__name__ for t in types)
#                                 raise TypeError(
#                                     f"Argument '{arg_name}' should be of type {expected_types_str}, "
#                                     f"but got {type(arg_value).__name__}"
#                                 )
#                         elif origin in (List, list):
#                             if not isinstance(arg_value, list):
#                                 raise TypeError(f"Argument '{arg_name}' should be of type list, but got {type(arg_value).__name__}")
#                             if expected_type.__args__ and not all(isinstance(item, expected_type.__args__[0]) for item in arg_value):
#                                     raise TypeError(f"Argument '{arg_name}' should be a list of {expected_type.__args__[0].__name__}")
#                         elif origin in (Tuple, tuple):
#                             if not isinstance(arg_value, tuple):
#                                  raise TypeError(f"Argument '{arg_name}' should be of type tuple, but got {type(arg_value).__name__}")
#                             if expected_type.__args__ and len(expected_type.__args__) != len(arg_value):
#                                 raise TypeError(f"Argument '{arg_name}' should be a tuple of length {len(expected_type.__args__)}")
#                             elif expected_type.__args__:
#                                 for i, expected_arg_type in enumerate(expected_type.__args__):
#                                     if not isinstance(arg_value[i], expected_arg_type):
#                                         raise TypeError(f"Argument '{arg_name}', element {i} should be of type {expected_arg_type}, but got {type(arg_value[i]).__name__}")
#                     else:
#                         raise ValueError(f"Type annotation for '{arg_name}' is not supported")
#             return func_to_wrap(*args, **kwargs)
#
#         return wrapper
#     return decorator
