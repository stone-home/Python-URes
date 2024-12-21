import datetime


def datetime_converter(
    time: datetime.datetime, iso8601: bool = True, format: str = "%Y%m%d-%H%M%S"
) -> str:
    """COnvert `datetime.datetime` object to iso8601 format or custom format.

    Args:
        time (datetime.datetime): the time want to be converted
        iso8601 (bool): if True, return iso8601 format. Defaults to True.
        format (str): custom format, and it is only available when iso8601 set to false. Defaults to "%Y%m%d-%H%M%S".

    Returns:
        str: the time in expected format
    """
    if iso8601:
        return time.isoformat() + "Z"
    else:
        return time.strftime(format)


def timestamp_converter(
    timestamp: int, iso8601: bool = True, format: str = "%Y%m%d-%H%M%S"
) -> str:
    """Convert unix timestamp to iso8601 format or custom format.

    Args:
        timestamp (int): a unix timestamp
        iso8601 (bool): if True, return iso8601 format. Defaults to True.
        format (str): custom format, and it is only available when iso8601 set to false. Defaults to "%Y%m%d-%H%M%S".

    Returns:
        str: the time in expected format
    """
    return datetime_converter(
        datetime.datetime.fromtimestamp(timestamp),
        iso8601=iso8601,
        format=format,
        # datetime.datetime.utcfromtimestamp(timestamp), iso8601=iso8601, format=format
    )


def time_now(iso8601: bool = True, format: str = "%Y%m%d-%H%M%S") -> str:
    """Generate a current time in iso8601 format or custom format.

    Args:
        iso8601 (bool): if True, return iso8601 format. Defaults to True.
        format (str): custom format, and it is only available when iso8601 set to false. Defaults to "%Y%m%d-%H%M%S".

    Returns:
        str: the time in expected format

    """
    return datetime_converter(
        datetime.datetime.now(datetime.UTC), iso8601=iso8601, format=format
    )
