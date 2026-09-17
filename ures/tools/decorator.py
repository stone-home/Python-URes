import logging
from collections.abc import Callable
from functools import wraps


logger = logging.getLogger(__name__)


def check_instance_variable(variable_name: str) -> Callable:
    """Skip a method when an instance attribute is missing or ``None``.

    Args:
        variable_name (str): Name of the instance attribute to require.

    Returns:
        A decorator. The wrapped method returns None when the attribute is
        missing or None.

    Examples:
        >>> from ures.tools.decorator import check_instance_variable
        >>> class Worker:
        ...     def __init__(self):
        ...         self.client = None
        ...     @check_instance_variable("client")
        ...     def ping(self):
        ...         return "ok"
        >>> Worker().ping() is None
        True
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
