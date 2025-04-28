from abc import abstractmethod, ABC
from ..type import MemoryBlock


class AbcAlgorithm(ABC):
    @abstractmethod
    def malloc(self, size: int, **kwargs) -> MemoryBlock:
        """
        Allocate memory on the device.
        """
        pass

    @abstractmethod
    def free(self, ptr: str, **kwargs) -> MemoryBlock:
        """
        Free memory on the device.
        """
        pass
