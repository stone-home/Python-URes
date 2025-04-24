from typing import Optional
import tqdm
from .manager import MemoryManager, MemoryBlock
from .allocators.pytorch.cuda_caching import CachingAllocator
from .allocators.interface import AllocatorInterface


class Simulator:
    def __init__(
            self,
            max_device_memory: int,
            allocator: Optional[AllocatorInterface]
    ):
        # The maximum device memory in bytes
        self._max_device_memory: int = max_device_memory
        _kwargs = {
            "max_memory": max_device_memory,
        }
        self._allocator: AllocatorInterface = allocator or CachingAllocator(**_kwargs)

    def simulate(self, memory_manager: MemoryManager):
        for blocks in tqdm.tqdm(memory_manager.memory.values()):
            for block in blocks:
                pass




