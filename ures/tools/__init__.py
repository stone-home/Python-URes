"""Small helpers such as enum inspection and instance-attribute guards.

Examples:
    >>> from enum import Enum
    >>> from ures.tools import EnumManipulator
    >>> class Color(Enum):
    ...     RED = 1
    >>> EnumManipulator(Color).fetch_keys() == ["RED"]
    True
"""

import os
import importlib
import pkgutil

package_dir = os.path.dirname(__file__)

for _, module_name, _ in pkgutil.iter_modules([package_dir]):
    module = importlib.import_module(f".{module_name}", package=__name__)

    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        if isinstance(attribute, type) and not attribute_name.startswith("_"):
            globals()[attribute_name] = attribute

__all__ = [name for name in globals() if not name.startswith("_")]
