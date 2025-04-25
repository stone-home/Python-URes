from abc import abstractmethod, ABC
from ..type import MemoryBlock


class AllocatorInterface(ABC):
    @abstractmethod
    def malloc(self, block: MemoryBlock, **kwargs):
        """
        Allocate memory on the device.
        """
        pass

    @abstractmethod
    def free(self, block: MemoryBlock, **kwargs):
        """
        Free memory on the device.
        """
        pass
