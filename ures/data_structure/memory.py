from abc import ABC, abstractmethod
from typing import Union, Optional


class AbsMemoryBlock(ABC):
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
    def type(self) -> str:
        return 'not used'

    @property
    def duration(self) -> Optional[Union[int, float]]:
        return (
            (self.free_time - self.alloc_time) if self.free_time is not None else None
        )

    @property
    def is_freed(self) -> bool:
        return self.free_time is not None

    def __repr__(self):
        return f"Memory({self.address}): {self.bytes} bytes, start: {self.alloc_time}, end: {self.free_time}, duration: {self.duration}"
