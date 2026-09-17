"""Memory-block simulation and pluggable allocation algorithms.

Examples:
    >>> from ures.memory import DeviceMemorySimulator
    >>> sim = DeviceMemorySimulator(device_id=0, total_memory=256)
    >>> sim.allocate(32).success
    True
"""
from .blocks import TraceInfo, MemoryBlock, MemoryInfo, Segment, BlockPool
from .allocator import (
    AllocationRequest,
    AllocationResult,
    AllocationStrategy,
    FreeResult,
    FreeRequest,
    MemoryAllocator,
    FirstFitAllocator,
    BestFitAllocator,
    WorstFitAllocator,
    NextFitAllocator,
    BuddySystemAllocator,
    DeviceMemorySimulator,
)

__all__ = [
    "TraceInfo",
    "MemoryBlock",
    "MemoryInfo",
    "Segment",
    "BlockPool",
    "AllocationRequest",
    "AllocationResult",
    "AllocationStrategy",
    "FreeResult",
    "FreeRequest",
    "MemoryAllocator",
    "FirstFitAllocator",
    "BestFitAllocator",
    "WorstFitAllocator",
    "NextFitAllocator",
    "BuddySystemAllocator",
    "DeviceMemorySimulator",
]
