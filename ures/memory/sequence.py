from collections import defaultdict
from typing import Generator, List, Tuple
from abc import ABC, abstractmethod
from enum import Enum
from .type import MemoryBlock


class AbsSequence(ABC):
    def __init__(self, memory_blocks: List[MemoryBlock]):
        self._memory_blocks: List[MemoryBlock] = memory_blocks

    @property
    def sequence(self) -> Generator[Tuple[int, MemoryBlock], None, None]:
        return self.preprocess()

    @abstractmethod
    def preprocess(self) -> Generator[Tuple[int, MemoryBlock], None, None]:
        pass


class MemoryOperations(Enum):
    ALLOCATE = 0
    FREE = 1


class MemorySequence(AbsSequence):
    def preprocess(self) -> Generator[Tuple[int, MemoryBlock], None, None]:
        time_based_blocks: dict[int, list[tuple[int, MemoryBlock]]] = defaultdict(list)
        for mb in self._memory_blocks:
            alloc_time = mb.alloc_time if mb.alloc_time is not None else 0
            # Due to time_based_blocks being a defaultdict, we do not need
            # to manually create a list when the key is not present
            time_based_blocks[alloc_time].append((MemoryOperations.ALLOCATE.value, mb))

            if mb.free_time is not None:
                time_based_blocks[mb.free_time].append((MemoryOperations.FREE.value, mb))

        # Yield operations sorted by timestamp
        # Iterate through timestamps in ascending order
        for timestamp in sorted(time_based_blocks.keys()):
            # Yield all operations scheduled for this timestamp
            # Using yield from simplifies yielding all elements from the inner list
            yield from time_based_blocks[timestamp]



