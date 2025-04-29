import copy
import logging
from typing import Optional, Dict, Union
from collections import OrderedDict
from .types import MemoryBlock
from .sequence import AbsSequence, MemorySequence
from ..tools.decorator import type_check


logger = logging.getLogger(__name__)


class MemoryManager:
    def __init__(self):
        self._memory: Dict[int, MemoryBlock] = OrderedDict()

    @property
    def memory(self) -> dict[int, MemoryBlock]:
        return self._memory

    @type_check
    def add(self, memory: MemoryBlock) -> None:
        self._memory[id(memory)] = memory
        logger.debug(f"Memory[{id(memory)}] added: {repr(memory)}")

    @type_check
    def remove(self, memory: MemoryBlock) -> Optional[MemoryBlock]:
        return self.remove_by_id(id(memory))

    @type_check
    def remove_by_id(self, memory_id: int) -> Optional[MemoryBlock]:
        if memory_id in self._memory.keys():
            _memory = self._memory.pop(memory_id)
            logger.debug(f"Memory[{memory_id}] removed")
            return _memory
        return None

    @type_check
    def exist(self, memory: MemoryBlock) -> bool:
        return id(memory) in self.memory.keys()

    def gen_sequence(
        self, sequence_processor: Optional[AbsSequence] = None
    ) -> AbsSequence:
        using_sequence = (
            MemorySequence if sequence_processor is None else sequence_processor
        )
        return using_sequence(memory_blocks=copy.deepcopy(list(self.memory.values())))
