import uuid


def zettelkasten_id() -> str:
    """Generate an id used for zettelkasten note-taking system only.

    Returns:
        str: a zettelkasten id

    """
    _id = uuid.uuid4().hex
    return f"{_id[:9]}.{_id[-11:]}"


def unique_id() -> str:
    """generate a unique id.

    Returns:
        str: a unique id

    """
    return uuid.uuid4().hex


def format_memory(nbytes: int) -> str:
    """Format memory size in human-readable format.

    Args:
        nbytes (int): the memory size in bytes

    Returns:
        str: the memory size in human-readable format

    **The function is copied from PyTorch source code.**
    """
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
