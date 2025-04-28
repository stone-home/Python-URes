from abc import ABC, abstractmethod
from typing import Optional
from ..algorithms.abc import AbcAlgorithm


class AbcDevice(ABC):
    """
    Interface for representing a hardware device that can have memory allocated to it.
    This interface defines the basic properties and operations that a memory-aware device should provide.
    """

    def __init__(self, algorithm: AbcAlgorithm):
        """
        Initializes the device with a specific memory allocation algorithm.

        Args:
            algorithm (AbcAlgorithm): The memory allocation algorithm to be used by the device.
        """
        self._algorithm = algorithm

    @property
    @abstractmethod
    def device_id(self) -> str:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name or identifier of the device. This should be a unique string.
        """
        pass

    @property
    @abstractmethod
    def total_memory(self) -> int:
        """
        The total memory capacity of the device in bytes.
        """
        pass

    @property
    @abstractmethod
    def available_memory(self) -> int:
        """
        The currently available memory on the device in bytes.
        """
        pass

    @abstractmethod
    def malloc(self, size: int) -> Optional[str]:
        """
        Allocates a block of memory of the specified size on the device.

        Args:
            size (int): The size of the memory block to allocate in bytes.

        Returns:
            Optional[str]: The starting address or a handle to the allocated memory block on the device,
                           or None if allocation fails.
        """
        pass

    @abstractmethod
    def free(self, ptr: str):
        """
        Deallocates the memory block at the specified address on the device.

        Args:
            ptr (str): The starting address or handle of the memory block to deallocate.
        """
        pass
