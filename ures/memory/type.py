import time
import logging
from abc import ABC, abstractmethod
from typing import Optional, Union
from ures.tools.decorator import type_check
from ures.data_structure.memory import MemoryBlockInterface, Memory
from ures.data_structure.bi_directional_links import NonCircularBiDirection
from ures.string import format_memory

logger = logging.getLogger(__name__)


# -------------- Classes are related with essential memory ------------------


class SegmentMemoryBlock(NonCircularBiDirection, MemoryBlockInterface):
    def __init__(self, memory: Memory):
        """
        Initialize a SegmentMemoryBlock with a Memory object.

        Args:
            memory (Memory): The Memory object to be associated with this block.
        """
        super().__init__(memory)
        self.is_allocated = False

    @property
    def memory(self) -> Memory:
        return self.value

    @property
    def bytes(self) -> int:
        return self.memory.bytes

    @property
    def address(self) -> str:
        return self.memory.address

    @property
    def alloc_time(self) -> Union[int, float]:
        return self.memory.alloc_time

    @property
    def free_time(self) -> Optional[Union[int, float]]:
        return self.memory.free_time

    @property
    def duration(self) -> Optional[Union[int, float]]:
        return self.free_time - self.alloc_time if self.free_time is not None else None

    @property
    def is_permanent(self) -> bool:
        return self.duration is None

    def splice(self, size: int) -> "SegmentMemoryBlock":
        """
        Splits this *free* memory block into two parts if possible.

        Creates a new block representing the first `size` bytes and returns it.
        The current block (`self`) is updated to represent the remaining bytes,
        with its address adjusted accordingly. The new block is inserted
        immediately before the current block in the linked list.

        If `size` equals the block's current size, the existing block's memory
        `alloc_time` is updated, and the block itself (`self`) is returned.
        The caller is expected to mark the returned block as allocated.

        Args:
            size (int): The number of bytes to allocate from the beginning
                        of this block.

        Returns:
            SegmentMemoryBlock: The new block representing the allocated segment,
                                or `self` if size matched the full block size.

        Raises:
            MemoryError: If the block is currently allocated (`is_allocated` is True).
            MemoryError: If the requested `size` is larger than the block's size.
            ValueError: If `size` is not positive.
        """
        if self.is_allocated:
            raise MemoryError(
                f"Cannot splice from an allocated memory block {self.address}. Please free it first."
            )
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"Splice size must be a positive integer, got {size}")
        if size > self.bytes:
            raise MemoryError(
                f"Cannot splice {size} bytes from {self.address}({self.bytes} bytes)."
            )

        logger.info(f"Splicing {size} bytes from {self.address}({self.bytes} bytes)...")
        current_address = self.address
        current_bytes = self.bytes
        alloc_t = time.time_ns()  # Record allocation time now

        if size == current_bytes:
            logger.debug(
                f"Splice size matches full block. Updating alloc_time for {self}"
            )
            # Update alloc_time on existing memory object
            self.memory.alloc_time = alloc_t
            # The caller will mark this block 'self' as allocated.
            return self
        else:
            # --- Create the new block for the requested size ---
            new_mem = Memory(
                bytes=size,
                address=current_address,
                alloc_time=alloc_t,
                free_time=None,  # It's being allocated now
            )
            new_block = SegmentMemoryBlock(new_mem)
            # new_block.is_allocated will be set by the caller

            # --- Update the current block (`self`) for the remaining part ---
            remaining_bytes = current_bytes - size
            remaining_address = current_address + size

            # Update the memory object within the current block
            # Option 1: Modify existing memory object (less clean)
            # self.memory.bytes = remaining_bytes
            # self.memory.address = remaining_address

            # Option 2: Create a new memory object for the remainder (cleaner)
            self._value = Memory(  # Replace self.memory via self.value
                bytes=remaining_bytes,
                address=remaining_address,
                alloc_time=self.alloc_time,
            )
            logger.debug(f"Original block updated: {self}")

            # --- Link the new block into the list ---
            # Insert new_block before the updated self
            self.insert_before(new_block)
            logger.debug(f"Inserted new block {new_block} before updated {self}")

            return new_block

    def coalesce(self) -> "SegmentMemoryBlock":
        if self.is_allocated is True:
            raise MemoryError(
                "Cannot coalesce an allocated memory block. Please free it first."
            )
        if self.prev is not None and self.prev.is_allocated is False:
            logger.info(f"start coalescing {self.prev} and {self}")
            _p_memory = self.prev
            self.prev.remove()
            self.memory.bytes += _p_memory.bytes
            self.memory.address = _p_memory.address
        if self.next is not None and self.next.is_allocated is False:
            logger.info(f"start coalescing {self} and {self.next}")
            _n_memory = self.next
            self.next.remove()
            self.memory.bytes += _n_memory.bytes

        # todo: maybe create another function to ensure all states are correct
        if self.alloc_time is not None:
            logger.warning(f"{self}'s alloc_time is not None, which should be None")
            # only freed memory block can be coalesced
            self.memory.alloc_time = None
        if self.free_time is not None:
            logger.warning(f"{self}'s free_time is not None, which should be None")
            # only freed memory block can be coalesced
            self.memory.free_time = None

        return self

    def __str__(self):
        return f"{self.address}({self.bytes} bytes)"


class Device(ABC):
    """
    Interface for representing a hardware device that can have memory allocated to it.
    This interface defines the basic properties and operations that a memory-aware device should provide.
    """

    def __init__(
        self,
        name: str,
        device_id: str,
        total_memory: int,
    ):
        """
        Initializes the device with a specific memory allocation algorithm.

        Args:
            algorithm (AbcAlgorithm): The memory allocation algorithm to be used by the device.
        """
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

    @abstractmethod
    def malloc(self, size: int, **kwargs):
        pass

    @abstractmethod
    def free(self, ptr: str, **kwargs):
        pass
