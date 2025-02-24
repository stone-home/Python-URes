import datetime


def datetime_converter(
    time: datetime.datetime, iso8601: bool = True, format: str = "%Y%m%d-%H%M%S"
) -> str:
    """
    Convert a datetime object to a formatted string.

    This function converts a given datetime.datetime object into a string. If 'iso8601' is True,
    the datetime is converted to ISO 8601 format with a trailing 'Z'. Otherwise, the datetime is
    formatted using the provided custom format string.

    Args:
        time (datetime.datetime): The datetime object to be converted.
        iso8601 (bool, optional): Determines the output format.
            - If True, returns the time in ISO 8601 format.
            - If False, returns the time formatted by the 'format' parameter.
            Defaults to True.
        format (str, optional): The custom format string to use when 'iso8601' is False.
            Defaults to "%Y%m%d-%H%M%S".

    Returns:
        str: The formatted time string.

    Example:
        >>> import datetime
        >>> dt = datetime.datetime(2020, 1, 1, 12, 0, 0)
        >>> datetime_converter(dt, iso8601=True)
        '2020-01-01T12:00:00Z'
        >>> datetime_converter(dt, iso8601=False, format="%Y/%m/%d %H:%M")
        '2020/01/01 12:00'
    """
    if iso8601:
        return time.isoformat() + "Z"
    else:
        return time.strftime(format)


def timestamp_converter(
    timestamp: int, iso8601: bool = True, format: str = "%Y%m%d-%H%M%S"
) -> str:
    """
    Convert a Unix timestamp to a formatted time string.

    This function converts a Unix timestamp (number of seconds since the epoch) into a human-readable
    string. It utilizes the 'datetime_converter' function to perform the conversion, allowing the output
    to be either in ISO 8601 format or in a custom format.

    Args:
        timestamp (int): The Unix timestamp to be converted.
        iso8601 (bool, optional): Determines the output format.
            - If True, returns the time in ISO 8601 format.
            - If False, returns the time formatted by the 'format' parameter.
            Defaults to True.
        format (str, optional): The custom format string to use when 'iso8601' is False.
            Defaults to "%Y%m%d-%H%M%S".

    Returns:
        str: The formatted time string.

    Example:
        >>> # Unix timestamp for 2020-01-01 12:00:00
        >>> timestamp_converter(1577880000, iso8601=True)
        '2020-01-01T12:00:00Z'
        >>> timestamp_converter(1577880000, iso8601=False, format="%d/%m/%Y %H:%M")
        '01/01/2020 12:00'
    """
    return datetime_converter(
        datetime.datetime.fromtimestamp(timestamp),
        iso8601=iso8601,
        format=format,
    )


def time_now(iso8601: bool = True, format: str = "%Y%m%d-%H%M%S") -> str:
    """
    Retrieve the current time as a formatted string.

    This function gets the current time using the system clock and converts it into a string.
    The output can either be in ISO 8601 format (with a trailing 'Z') or a custom format based on
    the provided format string.

    Args:
        iso8601 (bool, optional): Determines the output format.
            - If True, returns the time in ISO 8601 format.
            - If False, returns the time formatted by the 'format' parameter.
            Defaults to True.
        format (str, optional): The custom format string to use when 'iso8601' is False.
            Defaults to "%Y%m%d-%H%M%S".

    Returns:
        str: The current time as a formatted string.

    Example:
        >>> current_time = time_now(iso8601=True)
        >>> current_time.endswith("Z")
        True
        >>> time_now(iso8601=False, format="%H:%M:%S")
        '14:35:22'
    """
    import sys

    if sys.version_info[:2] == (3, 10):
        _time = datetime.datetime.now(datetime.timezone.utc)
    else:
        _time = datetime.datetime.now(datetime.UTC)
    return datetime_converter(_time, iso8601=iso8601, format=format)
