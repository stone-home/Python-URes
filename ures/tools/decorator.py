import logging
from functools import wraps


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
