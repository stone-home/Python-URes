import pytest
import time
from ures.memory.types import SegmentMemoryBlock
from ures.data_structure.memory import Memory


@pytest.fixture
def free_block_100_at_0() -> SegmentMemoryBlock:
    """Creates a single free block of 100 bytes at address 0."""
    # Give it an initial alloc_time to test coalesce reset
    mem = Memory(bytes=100, address=0, alloc_time=time.time_ns() - 1000)
    block = SegmentMemoryBlock(mem)
    block.is_allocated = False
    return block


@pytest.fixture
def free_block_50_at_100() -> SegmentMemoryBlock:
    """Creates a single free block of 50 bytes at address 100."""
    mem = Memory(bytes=50, address=100, alloc_time=time.time_ns() - 500)
    block = SegmentMemoryBlock(mem)
    block.is_allocated = False
    return block


@pytest.fixture
def free_block_200_at_150() -> SegmentMemoryBlock:
    """Creates a single free block of 200 bytes at address 150."""
    mem = Memory(bytes=200, address=150, alloc_time=time.time_ns() - 200)
    block = SegmentMemoryBlock(mem)
    block.is_allocated = False
    return block


@pytest.fixture
def allocated_block_75_at_50() -> SegmentMemoryBlock:
    """Creates a single allocated block of 75 bytes at address 50."""
    mem = Memory(bytes=75, address=50, alloc_time=time.time_ns() - 700)
    block = SegmentMemoryBlock(mem)
    block.is_allocated = True
    block.memory.free_time = None
    return block


@pytest.fixture
def linked_free_blocks(
    free_block_100_at_0, free_block_50_at_100, free_block_200_at_150
):
    """Creates a linked list of three free blocks: [0:100] <-> [100:50] <-> [150:200]"""
    b1 = free_block_100_at_0
    b2 = free_block_50_at_100
    b3 = free_block_200_at_150

    # Link them (direct assignment okay for setup)
    b1.insert_after(b2)
    b2.insert_after(b3)
    return b1, b2, b3
