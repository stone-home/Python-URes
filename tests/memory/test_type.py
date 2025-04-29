import time
import pytest
from ures.memory.types import SegmentMemoryBlock
from ures.data_structure.memory import Memory


class TestSegmentMemoryBlockInitAndProperties:
    """Tests for initialization and properties."""

    def test_init_success(self):
        """Verify basic initialization sets memory and defaults."""
        start_time = time.time_ns()
        mem = Memory(
            bytes=256, address=0x1000, alloc_time=start_time
        )  # Use hex address
        block = SegmentMemoryBlock(mem)

        assert block.memory is mem
        assert block.value is mem
        assert block.is_allocated is False
        assert block.prev is None
        assert block.next is None

    def test_property_delegation(self, allocated_block_75_at_50):
        """Verify properties delegate correctly to the underlying Memory object."""
        block = allocated_block_75_at_50
        mem = block.memory

        assert block.bytes == mem.bytes == 75
        assert block.address == mem.address == 50
        assert block.address_hex == hex(mem.address) == "0x32"  # Test hex property
        assert block.alloc_time == mem.alloc_time
        assert block.free_time is None
        mem.free_time = mem.alloc_time + 1000
        assert block.free_time == mem.free_time

    def test_duration_calculation(self, allocated_block_75_at_50):
        """Test duration calculation."""
        block = allocated_block_75_at_50
        assert block.duration is None
        block.memory.free_time = block.alloc_time + 5000
        assert block.duration == 5000

    def test_is_permanent(self, allocated_block_75_at_50):
        """Test is_permanent property."""
        block = allocated_block_75_at_50
        assert block.is_permanent is True
        block.memory.free_time = block.alloc_time + 5000
        assert block.is_permanent is False

    def test_str_representation(self, free_block_100_at_0):
        """Test the __str__ method uses address_hex."""
        block = free_block_100_at_0
        assert str(block) == f"{hex(0)}({100} bytes)"
        block.memory.address = 0xABC
        block.memory.bytes = 123
        assert str(block) == f"{hex(0xABC)}({123} bytes)"


class TestSegmentMemoryBlockSplice:
    """Tests for the splice() method."""

    def test_splice_allocated_raises_error(self, allocated_block_75_at_50):
        with pytest.raises(
            MemoryError, match="Cannot splice from an allocated memory block"
        ):
            allocated_block_75_at_50.splice(10)

    def test_splice_too_large_raises_error(self, free_block_100_at_0):
        with pytest.raises(MemoryError, match="Cannot splice 101 bytes"):
            free_block_100_at_0.splice(101)

    def test_splice_invalid_size_raises_error(self, free_block_100_at_0):
        with pytest.raises(ValueError, match="must be a positive"):
            free_block_100_at_0.splice(0)
        with pytest.raises(ValueError, match="must be a positive"):
            free_block_100_at_0.splice(-10)

    def test_splice_exact_size(self, free_block_100_at_0):
        block = free_block_100_at_0
        original_alloc_time = block.alloc_time
        returned_block = block.splice(100)

        assert returned_block is block
        assert block.bytes == 100
        assert block.address == 0
        assert block.alloc_time > original_alloc_time

    def test_splice_partial_size_updates_blocks(self, free_block_100_at_0):
        block = free_block_100_at_0
        original_alloc_time_segment = block.alloc_time

        splice_size = 40
        new_block = block.splice(splice_size)

        # Verify new block
        assert isinstance(new_block, SegmentMemoryBlock)
        assert new_block.address == 0
        assert new_block.bytes == splice_size
        assert new_block.is_allocated is False
        assert new_block.alloc_time is not None

        # Verify remaining block (variable 'block' points to remainder)
        assert block.address == splice_size
        assert block.bytes == 100 - splice_size
        assert block.is_allocated is False
        assert isinstance(block.value, Memory)  # Check Memory obj was replaced
        assert block.value.address == 40
        assert block.value.bytes == 60
        # Assuming remaining block keeps original segment alloc time
        assert block.alloc_time == original_alloc_time_segment

        # Verify list links using properties
        assert new_block.prev is None
        assert new_block.next is block
        assert block.prev is new_block
        assert block.next is None

    def test_splice_middle_block_links(self, linked_free_blocks):
        b1, b2, b3 = linked_free_blocks
        splice_size = 20
        new_block = b2.splice(splice_size)

        # Verify links using properties
        assert b1.next is new_block
        assert new_block.prev is b1
        assert new_block.next is b2
        assert b2.prev is new_block
        assert b2.next is b3
        assert b3.prev is b2


class TestSegmentMemoryBlockCoalesce:
    """Tests for the coalesce() method, including time resets."""

    def test_coalesce_allocated_raises_error(self, allocated_block_75_at_50):
        with pytest.raises(
            MemoryError, match="Cannot coalesce an allocated memory block"
        ):
            allocated_block_75_at_50.coalesce()

    def test_coalesce_no_neighbors_no_change(self, free_block_100_at_0):
        block = free_block_100_at_0
        original_bytes = block.bytes
        original_address = block.address
        original_alloc_time = block.alloc_time
        original_free_time = block.free_time

        returned_block = block.coalesce()

        assert returned_block is block
        assert block.bytes == original_bytes
        assert block.address == original_address
        assert block.alloc_time is None  # Should NOT be reset
        assert block.free_time is None  # Should NOT be reset
        assert block.prev is None
        assert block.next is None

    def test_coalesce_allocated_neighbors_no_change(self, free_block_50_at_100):
        # Create allocated neighbors using actual Memory
        b1 = SegmentMemoryBlock(Memory(bytes=20, address=80, alloc_time=time.time_ns()))
        b1.is_allocated = True
        b3 = SegmentMemoryBlock(
            Memory(bytes=30, address=150, alloc_time=time.time_ns())
        )
        b3.is_allocated = True

        b2 = free_block_50_at_100
        original_bytes = b2.bytes
        original_address = b2.address
        original_alloc_time = b2.alloc_time
        original_free_time = b2.free_time

        # Link them (direct assignment okay for setup)
        b1._next = b2
        b2._prev = b1
        b2._next = b3
        b3._prev = b2

        returned_block = b2.coalesce()

        assert returned_block is b2
        assert b2.bytes == original_bytes
        assert b2.address == original_address
        assert b2.alloc_time == None  # Should NOT be reset
        assert b2.free_time == None  # Should NOT be reset
        assert b2.prev is b1  # Check links via properties
        assert b2.next is b3

    def test_coalesce_backward_resets_times(self, linked_free_blocks):
        b1, b2, b3 = linked_free_blocks
        b1_address = b1.address  # Store original address
        b1_bytes = b1.bytes
        b2_bytes = b2.bytes
        assert b1.alloc_time is not None
        assert b2.alloc_time is not None

        returned_block = b2.coalesce()

        assert returned_block is b2
        assert b2.address == b1_address  # Check address updated correctly
        # Access b2.bytes directly as the memory object inside is modified
        assert b2.bytes == b1_bytes + b2_bytes + b3.bytes  # Size updated
        assert b2.is_allocated is False

        # Verify times are reset
        assert b2.alloc_time is None
        assert b2.free_time is None

        # Verify links via properties
        assert b2.prev is None
        assert b2.next is None
        assert b3.prev is None
        assert b1.next is None

    def test_coalesce_both_resets_times(self, linked_free_blocks):
        b1, b2, b3 = linked_free_blocks
        b1_address = b1.address
        total_bytes = b1.bytes + b2.bytes + b3.bytes
        assert b1.alloc_time is not None
        assert b2.alloc_time is not None
        assert b3.alloc_time is not None

        returned_block = b2.coalesce()

        assert returned_block is b2
        assert b2.address == b1_address  # Address should be b1's
        assert b2.bytes == total_bytes
        assert b2.is_allocated is False

        # Verify times are reset
        assert b2.alloc_time is None
        assert b2.free_time is None

        # Verify links via properties
        assert b2.prev is None
        assert b2.next is None
