# test_memory_sequence.py
import pytest
from typing import List
from ures.memory.type import MemoryBlock
from ures.memory.sequence import MemorySequence, AbsSequence, MemoryOperations
from .conftest import MockDevice


# --- Test Group: MemorySequence ---

class TestMemorySequence:
    """Tests for the MemorySequence class."""

    def test_empty_input(self):
        """Test MemorySequence with an empty list of memory blocks."""
        blocks: List[MemoryBlock] = []
        seq = MemorySequence(memory_blocks=blocks)
        # Consume the generator
        operations = list(seq.sequence)
        assert operations == []

    def test_single_block_alloc_only(self, mb_alloc_only: MemoryBlock):
        """Test with one block having only alloc_time."""
        seq = MemorySequence(memory_blocks=[mb_alloc_only])
        operations = list(seq.sequence)
        expected = [
            (MemoryOperations.ALLOCATE.value, mb_alloc_only)
        ]
        assert operations == expected

    def test_single_block_alloc_free(self, mb_alloc_free: MemoryBlock):
        """Test with one block having alloc and free times."""
        seq = MemorySequence(memory_blocks=[mb_alloc_free])
        operations = list(seq.sequence)
        # Expect alloc at time 5, free at time 15
        expected = [
            (MemoryOperations.ALLOCATE.value, mb_alloc_free), # time 5
            (MemoryOperations.FREE.value, mb_alloc_free)      # time 15
        ]
        assert operations == expected

    def test_single_block_alloc_time_none(self, mb_alloc_none: MemoryBlock):
        """Test block with alloc_time=None (maps to time 0)."""
        seq = MemorySequence(memory_blocks=[mb_alloc_none])
        operations = list(seq.sequence)
        # Expect alloc at time 0, free at time 8
        expected = [
            (MemoryOperations.ALLOCATE.value, mb_alloc_none), # time 0
            (MemoryOperations.FREE.value, mb_alloc_none)      # time 8
        ]
        assert operations == expected

    def test_single_block_free_time_none(self, mb_free_none: MemoryBlock):
        """Test block with free_time=None (only alloc operation generated)."""
        seq = MemorySequence(memory_blocks=[mb_free_none])
        operations = list(seq.sequence)
        # Expect alloc at time 3 only
        expected = [
            (MemoryOperations.ALLOCATE.value, mb_free_none) # time 3
        ]
        assert operations == expected

    def test_single_block_alloc_free_same_time(self, mb_alloc_free_same_time: MemoryBlock):
        """Test block with alloc and free at the same timestamp."""
        seq = MemorySequence(memory_blocks=[mb_alloc_free_same_time])
        operations = list(seq.sequence)
        # Both operations occur at time 12. The order within the same timestamp
        # depends on insertion order into the defaultdict list.
        # The code adds ALLOCATE then FREE if free_time is not None.
        expected = [
            (MemoryOperations.ALLOCATE.value, mb_alloc_free_same_time), # time 12
            (MemoryOperations.FREE.value, mb_alloc_free_same_time)      # time 12
        ]
        # Convert to sets for comparison if order within timestamp is not guaranteed/tested,
        # or assert the specific order if it is guaranteed.
        # assert set(operations) == set(expected) # Use if order at same time is not critical
        assert operations == expected # Use if order ALLOC->FREE at same time is expected

    def test_multiple_blocks_interleaved(self, mock_device: MockDevice):
        """Test multiple blocks with overlapping lifetimes."""
        mb1 = MemoryBlock(byte=10, device=mock_device, alloc_time=2, free_time=8)  # Lives 2-8
        mb2 = MemoryBlock(byte=20, device=mock_device, alloc_time=5, free_time=12) # Lives 5-12
        mb3 = MemoryBlock(byte=30, device=mock_device, alloc_time=10)              # Lives 10 onwards

        seq = MemorySequence(memory_blocks=[mb1, mb2, mb3]) # Order in list shouldn't matter
        operations = list(seq.sequence)

        expected = [
            (MemoryOperations.ALLOCATE.value, mb1), # time 2
            (MemoryOperations.ALLOCATE.value, mb2), # time 5
            (MemoryOperations.FREE.value, mb1),     # time 8
            (MemoryOperations.ALLOCATE.value, mb3), # time 10
            (MemoryOperations.FREE.value, mb2),     # time 12
        ]
        assert operations == expected

    def test_multiple_blocks_mixed_times(self, mock_device: MockDevice):
        """Test multiple blocks including None times and same timestamps."""
        mb_a = MemoryBlock(byte=1, device=mock_device, alloc_time=5, free_time=15) # 5-15
        mb_b = MemoryBlock(byte=2, device=mock_device, alloc_time=None, free_time=10) # 0-10
        mb_c = MemoryBlock(byte=3, device=mock_device, alloc_time=5)                # 5 onwards (no free)
        mb_d = MemoryBlock(byte=4, device=mock_device, alloc_time=10, free_time=10) # 10-10 (alloc/free same time)

        # Note: Order of blocks in the input list might affect order of operations
        # at the *same* timestamp if not explicitly sorted otherwise later.
        # The current implementation relies on list append order within defaultdict.
        seq = MemorySequence(memory_blocks=[mb_a, mb_b, mb_c, mb_d])
        operations = list(seq.sequence)

        # Expected sequence based on timestamps: 0, 5, 10, 15
        expected_ops_time_0 = [
            (MemoryOperations.ALLOCATE.value, mb_b),
        ]
        expected_ops_time_5 = [
            # Order depends on [mb_a, mb_b, mb_c, mb_d] input order -> mb_a then mb_c
            (MemoryOperations.ALLOCATE.value, mb_a),
            (MemoryOperations.ALLOCATE.value, mb_c),
        ]
        expected_ops_time_10 = [
            # Order depends on input order -> mb_b then mb_d
             (MemoryOperations.FREE.value, mb_b),
             (MemoryOperations.ALLOCATE.value, mb_d),
             (MemoryOperations.FREE.value, mb_d), # Alloc/Free for mb_d happen at same time 10
        ]
        expected_ops_time_15 = [
            (MemoryOperations.FREE.value, mb_a),
        ]

        expected_total = (
            expected_ops_time_0 +
            expected_ops_time_5 +
            expected_ops_time_10 +
            expected_ops_time_15
        )

        # For timestamps with multiple operations, check presence if order is not guaranteed.
        # Example check for time 5:
        ops_at_5 = [op for op in operations if op[0] == MemoryOperations.ALLOCATE.value and (op[1] == mb_a or op[1] == mb_c)]
        # assert len(ops_at_5) == 2
        # assert (MemoryOperations.ALLOCATE.value, mb_a) in ops_at_5
        # assert (MemoryOperations.ALLOCATE.value, mb_c) in ops_at_5

        # Check the full sequence if the order is deterministic and known:
        assert operations == expected_total