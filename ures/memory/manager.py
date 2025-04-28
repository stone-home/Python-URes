import time
from typing import Optional, Dict, List
from collections import OrderedDict
from ..tools.decorator import type_check
from .devices.interface import DeviceInterface
from .type import MemoryBlock


class MemoryManager:
    def __init__(self):
        pass


class SequenceGenerator:
    """
    A generator that yields a sequence of numbers starting from 0.
    """

    def __init__(self):
        self._current = 0

    def __iter__(self):
        return self

    def __next__(self):
        result = self._current
        self._current += 1
        return result
