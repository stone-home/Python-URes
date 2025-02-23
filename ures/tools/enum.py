from enum import Enum, EnumMeta
from typing import List, Optional


class EnumManipulator:
    """
    A class to manipulate and query an Enum object.

    This class provides helper methods to fetch keys, retrieve members and their values,
    check for key existence, and filter keys based on a keyword search.
    """

    def __init__(self, input_enum: EnumMeta):
        """
        Initialize the EnumManipulator instance with a given Enum.

        Args:
            input_enum (EnumMeta): An Enum class to be manipulated.

        Example:
            >>> from enum import Enum
            >>> class Color(Enum):
            ...     RED = 1
            ...     GREEN = 2
            >>> manipulator = EnumManipulator(Color)
        """
        self._enum = input_enum

    @property
    def fetch_enums(self) -> EnumMeta:
        """
        Retrieve the underlying Enum class.

        Returns:
            EnumMeta: The Enum class provided during initialization.

        Example:
            >>> from enum import Enum
            >>> class Color(Enum):
            ...     RED = 1
            ...     GREEN = 2
            >>> manipulator = EnumManipulator(Color)
            >>> enums = manipulator.fetch_enums
            >>> isinstance(enums, type)  # Enum classes are types
            True
        """
        return self._enum

    def fetch_keys(self) -> List[str]:
        """
        Get a list of all key names (member names) from the Enum.

        Returns:
            List[str]: A list of key names defined in the Enum.

        Example:
            >>> from enum import Enum
            >>> class Color(Enum):
            ...     RED = 1
            ...     BLUE = 3
            >>> manipulator = EnumManipulator(Color)
            >>> manipulator.fetch_keys()
            ['RED', 'BLUE']
        """
        return self.fetch_enums._member_names_

    def fetch_enum(self, key_name: str) -> Optional[Enum]:
        """
        Retrieve an Enum member by its key name (case-insensitive).

        Args:
            key_name (str): The key name to fetch.

        Returns:
            Optional[Enum]: The Enum member if found; otherwise, None.

        Example:
            >>> from enum import Enum
            >>> class Color(Enum):
            ...     RED = 1
            ...     GREEN = 2
            >>> manipulator = EnumManipulator(Color)
            >>> member = manipulator.fetch_enum("red")
            >>> member.value
            1
        """
        for _key in self.fetch_keys():
            if key_name.lower() == str(_key).lower():
                return self.fetch_enums[_key]
        return None

    def check_key(self, key_name: str) -> bool:
        """
        Check whether a given key exists in the Enum.

        Args:
            key_name (str): The key name to check.

        Returns:
            bool: True if the key exists; otherwise, False.

        Example:
            >>> from enum import Enum
            >>> class Color(Enum):
            ...     RED = 1
            ...     BLUE = 3
            >>> manipulator = EnumManipulator(Color)
            >>> manipulator.check_key("RED")
            True
            >>> manipulator.check_key("GREEN")
            False
        """
        return self.fetch_enum(key_name) is not None

    def fetch_value(self, key_name: str):
        """
        Retrieve the value associated with a given key in the Enum.

        Args:
            key_name (str): The key name for which to fetch the value.

        Returns:
            any: The value corresponding to the key if found; otherwise, None.

        Example:
            >>> from enum import Enum
            >>> class Status(Enum):
            ...     SUCCESS = "ok"
            ...     FAILURE = "error"
            >>> manipulator = EnumManipulator(Status)
            >>> manipulator.fetch_value("FAILURE")
            'error'
        """
        member = self.fetch_enum(key_name)
        if member is not None:
            return member.value
        return None

    def filter_by(self, keyword: str, field: str = None) -> list:
        """
        Filter and return keys from the Enum where the specified keyword matches.

        If no field is provided, the keyword is compared to the string representation
        of the Enum member's value. Otherwise, the specified attribute (field) of the member's
        value is used for comparison.

        Args:
            keyword (str): The keyword to search for.
            field (str, optional): The attribute of the Enum member's value to search within.
                Defaults to None.

        Returns:
            list: A list of keys for which the keyword was found.

        Example:
            >>> from enum import Enum
            >>> class Fruit(Enum):
            ...     APPLE = "red"
            ...     BANANA = "yellow"
            ...     GRAPE = "purple"
            >>> manipulator = EnumManipulator(Fruit)
            >>> manipulator.filter_by("red")
            ['APPLE']
        """
        result = []
        for key in self.fetch_keys():
            member_value = self.fetch_enums[key].value
            if field is None:
                if keyword == str(member_value):
                    result.append(key)
            else:
                # Assume the member_value has an attribute named field.
                field_value = getattr(member_value, field, None)
                if isinstance(field_value, str):
                    if keyword.lower() in field_value.lower():
                        result.append(key)
                elif isinstance(field_value, list):
                    if keyword.lower() in [str(item).lower() for item in field_value]:
                        result.append(key)
        return result
