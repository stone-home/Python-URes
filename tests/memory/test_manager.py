import time
import pytest
from collections import OrderedDict
from typing import Optional
from ures.memory.manager import MemoryManager, MemoryBlock

class ConcreteMemoryBlock(MemoryBlock):
    def __init__(self, address: int, size: int, comment: str = ""):
        self._address = address
        self._bytes = bytearray(size)
        self._comment = comment
        self._alloc_time = time.time_ns()
        self._free_time: Optional[int] = None

    @property
    def address(self) -> int:
        return self._address

    @property
    def bytes(self) -> bytearray:
        return self._bytes

    @property
    def alloc_time(self) -> int:
        return self._alloc_time

    @property
    def free_time(self) -> Optional[int]:
        return self._free_time

    @free_time.setter
    def free_time(self, value: Optional[int]):
        self._free_time = value

    @property
    def comment(self) -> str:
        return self._comment


class TestMemoryManagerInitialization:
    def test_initialization(self):
        manager = MemoryManager()
        assert isinstance(manager.memory, dict)
        assert not manager.memory
        assert manager._history is None

class TestMemoryManagerAddOperation:
    def test_add_memory_block(self):
        manager = MemoryManager()
        memory_block = ConcreteMemoryBlock(0, 10, "test")
        manager.add(memory_block)
        assert id(memory_block) in manager.memory
        assert manager.memory[id(memory_block)] is memory_block
        assert manager._history is not None
        assert len(manager._history) == 1
        _, record = manager._history.popitem()
        assert record.startswith("add|")
        assert repr(memory_block) in record

    def test_add_multiple_memory_blocks(self):
        manager = MemoryManager()
        memory_block1 = ConcreteMemoryBlock(0, 10, "test1")
        memory_block2 = ConcreteMemoryBlock(10, 20, "test2")
        manager.add(memory_block1)
        manager.add(memory_block2)
        assert id(memory_block1) in manager.memory
        assert id(memory_block2) in manager.memory
        assert len(manager._history) == 2

    def test_add_invalid_type(self):
        manager = MemoryManager()
        with pytest.raises(TypeError):
            manager.add("not a memory block") # type: ignore

class TestMemoryManagerRemoveOperation:
    def test_remove_existing_memory_block(self):
        manager = MemoryManager()
        memory_block = ConcreteMemoryBlock(0, 10, "test")
        manager.add(memory_block)
        manager.remove(memory_block)
        assert id(memory_block) not in manager.memory
        assert manager._history is not None
        assert len(manager._history) == 2
        records = list(manager._history.values())
        assert records[1].startswith("remove|")
        assert repr(memory_block) in records[1]

    def test_remove_nonexistent_memory_block(self):
        manager = MemoryManager()
        memory_block = ConcreteMemoryBlock(0, 10, "test")
        manager.remove(memory_block) # Should not raise an error
        assert id(memory_block) not in manager.memory
        assert manager._history is None # History is only created on add/remove

    def test_remove_invalid_type(self):
        manager = MemoryManager()
        with pytest.raises(TypeError):
            manager.remove("not a memory block") # type: ignore

class TestMemoryManagerExistOperation:
    def test_exist_existing_memory_block(self):
        manager = MemoryManager()
        memory_block = ConcreteMemoryBlock(0, 10, "test")
        manager.add(memory_block)
        assert manager.exist(memory_block)

    def test_exist_nonexistent_memory_block(self):
        manager = MemoryManager()
        memory_block = ConcreteMemoryBlock(0, 10, "test")
        assert not manager.exist(memory_block)

    def test_exist_after_remove(self):
        manager = MemoryManager()
        memory_block = ConcreteMemoryBlock(0, 10, "test")
        manager.add(memory_block)
        manager.remove(memory_block)
        assert not manager.exist(memory_block)

    def test_exist_invalid_type(self):
        manager = MemoryManager()
        with pytest.raises(TypeError):
            manager.exist("not a memory block") # type: ignore

class TestMemoryManagerHistoryOperation:
    def test_history_empty(self):
        manager = MemoryManager()
        assert manager.history() is None

    def test_history_after_add(self):
        manager = MemoryManager()
        memory_block = ConcreteMemoryBlock(0, 10, "test")
        manager.add(memory_block)
        history = manager.history()
        assert isinstance(history, OrderedDict)
        assert len(history) == 1
        _, record = history.popitem()
        assert record.startswith("add|")
        assert repr(memory_block) in record

    def test_history_after_add_and_remove(self):
        manager = MemoryManager()
        memory_block = ConcreteMemoryBlock(0, 10, "test")
        manager.add(memory_block)
        manager.remove(memory_block)
        history = manager.history()
        assert isinstance(history, OrderedDict)
        assert len(history) == 2
        records = list(history.values())
        assert records[0].startswith("add|")
        assert records[1].startswith("remove|")
        assert repr(memory_block) in records[0] and repr(memory_block) in records[1]

class TestMemoryManagerCleanHistoryOperation:
    def test_clean_history_when_history_exists(self):
        manager = MemoryManager()
        memory_block = ConcreteMemoryBlock(0, 10, "test")
        manager.add(memory_block)
        assert manager._history is not None
        manager.clean_history()
        assert manager._history is None

    def test_clean_history_when_history_is_none(self):
        manager = MemoryManager()
        manager.clean_history() # Should not raise an error
        assert manager._history is None


class TestMemoryManagerTimeBasedSequence:
    def test_empty_memory(self):
        manager = MemoryManager()
        result = manager.get_time_based_sequence()
        assert result == {}

    def test_single_allocation(self):
        manager = MemoryManager()
        mb1 = ConcreteMemoryBlock(0, 10, "block1")
        manager.add(mb1)
        result = manager.get_time_based_sequence()
        assert len(result) == 1
        assert list(result.values())[0] == [mb1]

    def test_allocation_and_free(self):
        manager = MemoryManager()
        mb1 = ConcreteMemoryBlock(0, 10, "block1")
        manager.add(mb1)
        free_time = time.time_ns()
        mb1.free_time = free_time
        result = manager.get_time_based_sequence()
        assert len(result) == 2 # Changed from 2 to 1.  The block is only in the list once.
        assert result[mb1.alloc_time] == [mb1] #  The block will be at its alloc time

    def test_multiple_allocations(self):
        manager = MemoryManager()
        mb1 = ConcreteMemoryBlock(0, 10, "block1")
        mb2 = ConcreteMemoryBlock(10, 20, "block2")
        manager.add(mb1)
        manager.add(mb2)
        result = manager.get_time_based_sequence()
        assert len(result) == 2
        assert result[mb1.alloc_time] == [mb1]
        assert result[mb2.alloc_time] == [mb2]

    def test_mixed_allocations_and_frees(self):
        manager = MemoryManager()
        mb1 = ConcreteMemoryBlock(0, 10, "block1")
        mb2 = ConcreteMemoryBlock(10, 20, "block2")
        manager.add(mb1)
        manager.add(mb2)
        free_time_1 = time.time_ns()
        mb1.free_time = free_time_1
        free_time_2 = time.time_ns()
        mb2.free_time = free_time_2
        result = manager.get_time_based_sequence()
        assert len(result) == 4 # Changed from 4
        assert result[mb1.alloc_time] == [mb1]
        assert result[mb2.alloc_time] == [mb2]

    def test_same_time_allocations(self):
        manager = MemoryManager()
        now = time.time_ns()
        mb1 = ConcreteMemoryBlock(0, 10, "block1")
        mb1._alloc_time = now
        mb2 = ConcreteMemoryBlock(10, 20, "block2")
        mb2._alloc_time = now
        manager.add(mb1)
        manager.add(mb2)
        result = manager.get_time_based_sequence()
        assert len(result) == 1
        assert result[now] == [mb1, mb2]

    def test_memory_reuse(self):
        manager = MemoryManager()
        mb1 = ConcreteMemoryBlock(0, 10, "block1")
        manager.add(mb1)
        free_time_1 = time.time_ns()
        mb1.free_time = free_time_1

        mb2 = ConcreteMemoryBlock(0, 10, "block2") #same address as mb1
        mb2._alloc_time = free_time_1 + 10  # Allocate after mb1 is freed.
        manager.add(mb2)
        result = manager.get_time_based_sequence()
        # Memory Block 1 has allocation and free operation
        # Memory Block 2 has allocation operation
        # So the result should be 3
        assert len(result) == 3
        assert result[mb1.alloc_time] == [mb1]
        assert result[mb1.free_time] == [mb1]
        assert result[mb2.alloc_time] == [mb2]

