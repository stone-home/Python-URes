import time
from typing import Optional, Union
from ures.tools.decorator import type_check
from ures.data_structure.memory import AbsMemoryBlock
from ures.string import format_memory
from .devices.interface import DeviceInterface


class MemoryBlock(AbsMemoryBlock):
    def __init__(self, byte: int, device: DeviceInterface, **kwargs):
        self._request_bytes: int = byte
        self._device: DeviceInterface = device
        self._actual_bytes: Optional[int] = kwargs.get("actual_bytes", None)
        self._address: Optional[str] = kwargs.get("address", None)
        self._alloc_time: Optional[Union[int, float]] = kwargs.get("alloc_time", None)
        self._free_time: Optional[Union[int, float]] = kwargs.get("free_time", None)
        self._comment: str = kwargs.get("comment", "")

    @property
    def bytes(self) -> int:
        return self._request_bytes

    @property
    def allocated_size(self) -> Optional[int]:
        return self._actual_bytes

    @property
    def address(self) -> Optional[str]:
        return self._address

    @address.setter
    @type_check
    def address(self, value: Optional[str]):
        self._address = value

    @property
    def alloc_time(self) -> Optional[Union[int, float]]:
        return self._alloc_time

    @alloc_time.setter
    @type_check
    def alloc_time(self, value: Optional[Union[int, float]]):
        if value is None:
            self._alloc_time = time.time_ns()
        else:
            self._alloc_time = value

    @property
    def free_time(self) -> Optional[Union[int, float]]:
        return self._free_time

    @free_time.setter
    @type_check
    def free_time(self, value: Optional[Union[int, float]]):
        if value is None:
            self._free_time = time.time_ns()
        else:
            self._free_time = value

    @property
    def comment(self) -> str:
        return self._comment

    @comment.setter
    @type_check
    def comment(self, value: str):
        self._comment = value

    @property
    def device(self) -> DeviceInterface:
        return self._device

    @device.setter
    @type_check
    def device(self, device: DeviceInterface):
        self._device = device

    def __repr__(self):
        return f"{self.device.name}|{self.address}|{self.comment}|{self.bytes}|{self.alloc_time}|{self.free_time}"

    def __str__(self):
        return f"{format_memory(self.bytes)} in Address {self.address} ({self.comment})"
