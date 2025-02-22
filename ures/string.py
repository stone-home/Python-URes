import uuid


def zettelkasten_id() -> str:
    """
    Generate a Zettelkasten identifier.

    This function creates a unique identifier suitable for a Zettelkasten note-taking system.
    It generates a UUID, then formats the hexadecimal string by taking the first 9 characters,
    appending a dot, and then the last 11 characters of the UUID.

    Returns:
        str: A formatted unique identifier (e.g., "abc123def.ghi45678901").

    Example:
        >>> id_str = zettelkasten_id()
        >>> isinstance(id_str, str)
        True
    """
    _id = uuid.uuid4().hex
    return f"{_id[:9]}.{_id[-11:]}"


def unique_id() -> str:
    """
    Generate a unique identifier.

    This function returns a unique identifier as a hexadecimal string generated from a UUID.

    Returns:
        str: A unique hexadecimal identifier (e.g., "3f8a7c2d1e9b4a6f8d0e2c1b3a5f7e9d").

    Example:
        >>> uid = unique_id()
        >>> len(uid) == 32
        True
    """
    return uuid.uuid4().hex


def format_memory(nbytes: int) -> str:
    """
    Format a memory size into a human-readable string.

    Converts a memory size in bytes into a formatted string using appropriate units (B, KB, MB, or GB)
    with two decimal places of precision. If the provided value is None, it returns "0 bytes".

    Args:
        nbytes (int): The memory size in bytes.

    Returns:
        str: The memory size in a human-readable format.

    Example:
        >>> format_memory(1024)
        '1.00 KB'
        >>> format_memory(1048576)
        '1.00 MB'
    """
    if nbytes is None:
        return "0 bytes"

    KB = 1024
    MB = 1024 * KB
    GB = 1024 * MB
    if abs(nbytes) >= GB:
        return f"{nbytes * 1.0 / GB:.2f} GB"
    elif abs(nbytes) >= MB:
        return f"{nbytes * 1.0 / MB:.2f} MB"
    elif abs(nbytes) >= KB:
        return f"{nbytes * 1.0 / KB:.2f} KB"
    else:
        return str(nbytes) + " B"


def capitalize_string(string: str, separator: str = " ") -> str:
    """
    Capitalize each word in a string using a specified separator.

    Splits the input string by the given separator, capitalizes the first letter of each part,
    and then joins the parts back together using the same separator.

    Args:
        string (str): The string to be capitalized.
        separator (str, optional): The separator to use for splitting and joining the string.
                                   Defaults to " ".

    Returns:
        str: The capitalized string.

    Example:
        >>> capitalize_string("hello world")
        'Hello World'
        >>> capitalize_string("john-doe", separator="-")
        'John-Doe'
    """
    return separator.join([item.capitalize() for item in string.split(separator)])
