import time
import logging
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from typing import Optional, Union
from ures.tools.decorator import type_check
from ures.data_structure.memory import AbsMemoryBlock
from ures.string import format_memory

logger = logging.getLogger(__name__)


# -------------- Classes are related with essential memory ------------------


class DeviceMemoryBlock(BaseModel):
    request_bytes: int = Field(
        ..., description="The requested size of the memory block in bytes."
    )
    actual_bytes: Optional[int] = Field(
        None, description="The actual allocated size of the memory block in bytes."
    )
    address: Optional[str] = Field(None, description="The address of the memory block.")
    alloc_time: Optional[Union[int, float]] = Field(
        None, description="The time when the memory block was allocated."
    )
    free_time: Optional[Union[int, float]] = Field(
        None, description="The time when the memory block was freed."
    )
    comment: str = Field(
        "", description="A comment or description for the memory block."
    )


class SegmentBlock(DeviceMemoryBlock):
    allocated_blocks: list[DeviceMemoryBlock] = Field(
        default=[], description="List of allocated memory blocks."
    )


class AbcMemoryAlgorithm(ABC):
    @abstractmethod
    def malloc(self, size: int, **kwargs) -> Optional[DeviceMemoryBlock]:
        """Allocate memory of size bytes and return a MemoryBlock object."""
        pass

    @abstractmethod
    def free(self, ptr, **kwargs) -> Optional[DeviceMemoryBlock]:
        """Free the memory block."""
        pass


class Device(AbcMemoryAlgorithm):
    """
    Interface for representing a hardware device that can have memory allocated to it.
    This interface defines the basic properties and operations that a memory-aware device should provide.
    """

    def __init__(
        self,
        algorithm: AbcMemoryAlgorithm,
        name: str,
        device_id: str,
        total_memory: int,
    ):
        """
        Initializes the device with a specific memory allocation algorithm.

        Args:
            algorithm (AbcAlgorithm): The memory allocation algorithm to be used by the device.
        """
        self._algorithm: AbcMemoryAlgorithm = algorithm
        self._name: str = name
        self._device_id: str = device_id
        self._total_memory: int = total_memory
        self._available_memory: int = total_memory
        self._allocated_memory: int = 0

    @property
    def device_id(self) -> str:
        return self._device_id

    @property
    def name(self) -> str:
        """
        The name or identifier of the device. This should be a unique string.
        """
        return self._name

    @property
    def total_memory(self) -> int:
        """
        The total memory capacity of the device in bytes.
        """
        return self._total_memory

    @property
    def available_memory(self) -> int:
        """
        The currently available memory on the device in bytes.
        """
        return self._available_memory

    @property
    def allocated_memory(self) -> int:
        """
        The currently allocated memory on the device in bytes.
        """
        return self._allocated_memory

    def malloc(self, size: int, **kwargs) -> Optional[DeviceMemoryBlock]:
        """
        Allocates a block of memory of the specified size on the device.

        Args:
            size (int): The size of the memory block to allocate in bytes.

        Returns:
            Optional[str]: The starting address or a handle to the allocated memory block on the device,
                           or None if allocation fails.
        """
        memory_block = self._algorithm.malloc(size)
        if memory_block is None:
            logger.error(f"Memory allocation failed: {size} bytes")
        else:
            self._allocated_memory += memory_block.allocated_size
            self._available_memory -= memory_block.allocated_size
        return memory_block

    def free(self, ptr: str, **kwargs) -> Optional[DeviceMemoryBlock]:
        """
        Deallocates the memory block at the specified address on the device.

        Args:
            ptr (str): The starting address or handle of the memory block to deallocate.
        """
        memory_block = self._algorithm.free(ptr)
        if memory_block is None:
            logger.error(f"Memory deallocation failed: {ptr}")
        else:
            self._allocated_memory -= memory_block.allocated_size
            self._available_memory += memory_block.allocated_size
        return memory_block
