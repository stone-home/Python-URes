import copy
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, List
from collections import OrderedDict
from ures.data_structure.memory import AbsMemoryBlock
from ures.string import format_memory
from ures.tools.decorator import type_check


class MemoryBlock(AbsMemoryBlock, ABC):
    @property
    @abstractmethod
    def comment(self) -> str:
        pass

    def __repr__(self):
        return f"{self.address}|{self.comment}|{self.bytes}|{self.alloc_time}|{self.free_time}"

    def __str__(self):
        return f"{format_memory(self.bytes)} in Address {self.address} ({self.comment})"


class MemoryManager:
    def __init__(self):
        self._memory: dict[int, MemoryBlock] = {}
        self._history: Optional[OrderedDict[int, str]] = None

    @property
    def memory(self) -> dict[int, MemoryBlock]:
        return self._memory

    @type_check
    def add(self, memory: MemoryBlock) -> None:
        self._memory[id(memory)] = memory
        self.__record_history(memory, "add")

    @type_check
    def remove(self, memory: MemoryBlock) -> None:
        mem_id = id(memory)
        if mem_id in self._memory.keys():
            del self._memory[mem_id]
            self.__record_history(memory, "remove")

    @type_check
    def exist(self, memory: MemoryBlock) -> bool:
        return id(memory) in self.memory.keys()

    def __record_history(self, memory: MemoryBlock, operation: str) -> None:
        record_str = f"{operation}|{repr(memory)}"
        if self._history is None:
            self._history = OrderedDict()
        self._history[time.time_ns()] = record_str

    def history(self) -> OrderedDict[int, str]:
        return self._history

    def clean_history(self) -> None:
        if self._history is not None:
            self._history.clear()
            self._history = None

    def get_time_based_sequence(self) -> Dict[int, List[MemoryBlock]]:
        """
        Retrieves memory blocks ordered by allocation and deallocation times.

        Returns:
            Dict[int, List[MemoryBlock]]: A dictionary where keys are timestamps
                (allocation or deallocation times) and values are lists of memory
                blocks active at that time.  A MemoryBlock will appear twice if
                it is allocated and freed.
        """
        time_based_blocks: Dict[int, List[MemoryBlock]] = {}
        for mb in self.memory.values():  # Iterate through the original memory dictionary
            alloc_time = mb.alloc_time
            free_time = mb.free_time

            if alloc_time not in time_based_blocks:
                time_based_blocks[alloc_time] = []
            time_based_blocks[alloc_time].append(mb)

            if free_time is not None:
                if free_time not in time_based_blocks:
                    time_based_blocks[free_time] = []
                time_based_blocks[free_time].append(mb)

        return dict(sorted(time_based_blocks.items()))



