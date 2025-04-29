from warnings import warn
from abc import ABC, abstractmethod
from typing import Union, Optional
from pydantic import BaseModel, Field


class Memory(BaseModel):
    bytes: int = Field(..., description="The size of the memory block in bytes.")
    address: Optional[str] = Field(
        None, description="The starting address of the memory block."
    )
    alloc_time: Optional[Union[int, float]] = Field(
        None, description="The time when the memory block was allocated."
    )
    free_time: Optional[Union[int, float]] = Field(
        None, description="The time when the memory block was freed."
    )

    @property
    def duration(self) -> Optional[Union[int, float]]:
        """
        Calculate the duration of the memory block's allocation.

        Returns:
            Optional[Union[int, float]]: The duration in seconds, or None if the memory is still allocated.
        """
        return (
            (self.free_time - self.alloc_time) if self.free_time is not None else None
        )

    @property
    def is_permanent(self) -> bool:
        """
        Check if the memory block is permanent (not freed).

        Returns:
            bool: True if the memory block is permanent, False otherwise.
        """
        return self.free_time is None

    def __repr__(self):
        return f"Memory({self.address})|{self.bytes} bytes|{self.alloc_time}->{self.free_time}|dur: {self.duration}"


class AbsMemoryBlock(ABC):
    def __init__(self):
        warn(
            f"This class({self.__class__.__name__}) will be removed in future versions. Using Memory class instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    @property
    @abstractmethod
    def bytes(self) -> int:
        pass

    @property
    @abstractmethod
    def address(self) -> str:
        pass

    @property
    @abstractmethod
    def alloc_time(self) -> Union[int, float]:
        pass

    @property
    @abstractmethod
    def free_time(self) -> Optional[Union[int, float]]:
        pass

    @property
    def duration(self) -> Optional[Union[int, float]]:
        return (
            (self.free_time - self.alloc_time) if self.free_time is not None else None
        )

    @property
    def is_freed(self) -> bool:
        return self.free_time is not None

    def __repr__(self):
        return f"Memory({self.address})|{self.bytes} bytes|{self.alloc_time}->{self.free_time}|dur: {self.duration}"
