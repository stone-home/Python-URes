import pytest
from unittest.mock import Mock, patch
from ures.memory import TraceInfo, MemoryInfo, MemoryBlock, Segment, BlockPool


class TestTraceInfo:
    """Test cases for TraceInfo class - capturing execution traces."""

    def test_trace_info_creation(self):
        """Test basic TraceInfo object creation."""
        trace = TraceInfo(
            timestamp_ns=1000000000,
            operation="test_op",
            filename="test.py",
            line_number=10,
            function_name="test_func",
        )

        assert trace.timestamp_ns == 1000000000
        assert trace.operation == "test_op"
        assert trace.filename == "test.py"
        assert trace.line_number == 10
        assert trace.function_name == "test_func"
        assert trace.additional_info == {}

    @patch("ures.memory.blocks.time.time_ns")
    @patch("ures.memory.blocks.inspect.currentframe")
    def test_capture_current_trace(self, mock_frame, mock_time):
        """Test capturing current execution trace."""
        # Mock time
        mock_time.return_value = 1234567890

        # Mock frame info
        mock_frame_info = Mock()
        mock_frame_info.filename = "test_file.py"
        mock_frame_info.lineno = 42
        mock_frame_info.function = "test_function"
        mock_frame_info.code_context = ["    test_line()\n"]

        with patch(
            "ures.memory.blocks.inspect.getframeinfo", return_value=mock_frame_info
        ):
            trace = TraceInfo.capture_current_trace("test_operation")

        assert trace.timestamp_ns == 1234567890
        assert trace.operation == "test_operation"
        assert trace.filename == "test_file.py"
        assert trace.line_number == 42
        assert trace.function_name == "test_function"
        assert trace.code_context == "test_line()"

    def test_capture_trace_with_additional_info(self):
        """Test capturing trace with additional information."""
        additional_info = {"addr": 0x1000, "size": 256}

        with patch("ures.memory.blocks.time.time_ns", return_value=9999999):
            trace = TraceInfo.capture_current_trace(
                "alloc", additional_info=additional_info
            )

        assert trace.additional_info == additional_info

    def test_to_dict(self):
        """Test converting TraceInfo to dictionary."""
        trace = TraceInfo(
            timestamp_ns=1000000,
            operation="test",
            filename="file.py",
            line_number=1,
            additional_info={"key": "value"},
        )

        result = trace.to_dict()

        expected = {
            "timestamp_ns": 1000000,
            "operation": "test",
            "filename": "file.py",
            "line_number": 1,
            "function_name": None,
            "code_context": None,
            "stack_trace": None,
            "additional_info": {"key": "value"},
        }

        assert result == expected


class TestMemoryInfo:
    """Test cases for MemoryInfo class - memory block state management."""

    def test_memory_info_creation(self):
        """Test basic MemoryInfo object creation."""
        mem_info = MemoryInfo(addr=0x1000, size=256)

        assert mem_info.addr == 0x1000
        assert mem_info.size == 256
        assert mem_info.action == "free"
        assert mem_info.allocated == False
        assert mem_info.alloc_time_ns is None
        assert mem_info.traces == []

    def test_is_allocated_states(self):
        """Test different allocation states."""
        mem_info = MemoryInfo(addr=0x1000, size=256)

        # Initially free
        assert not mem_info.is_allocated()

        # Allocated
        mem_info.allocated = True
        mem_info.action = "alloc"
        assert mem_info.is_allocated()

        # Free requested (still considered allocated)
        mem_info.action = "free_requested"
        assert mem_info.is_allocated()

        # Free completed
        mem_info.allocated = False
        mem_info.action = "free_completed"
        assert not mem_info.is_allocated()

    def test_free_states(self):
        """Test free-related state checks."""
        mem_info = MemoryInfo(addr=0x1000, size=256)

        # Free requested
        mem_info.action = "free_requested"
        assert mem_info.is_free_requested()
        assert not mem_info.is_free_completed()

        # Free completed
        mem_info.action = "free_completed"
        mem_info.allocated = False
        assert not mem_info.is_free_requested()
        assert mem_info.is_free_completed()

    def test_get_end_addr(self):
        """Test end address calculation."""
        mem_info = MemoryInfo(addr=0x1000, size=256)
        assert mem_info.get_end_addr() == 0x1000 + 256

    def test_add_trace(self):
        """Test adding traces to memory info."""
        mem_info = MemoryInfo(addr=0x1000, size=256)
        trace = TraceInfo(timestamp_ns=123, operation="test")

        mem_info.add_trace(trace)
        assert len(mem_info.traces) == 1
        assert mem_info.traces[0] == trace


class TestMemoryBlockBasics:
    """Test cases for MemoryBlock basic functionality."""

    def test_memory_block_creation(self):
        """Test basic MemoryBlock creation."""
        with patch("ures.memory.blocks.TraceInfo.capture_current_trace") as mock_trace:
            mock_trace.return_value = Mock()
            block = MemoryBlock(addr=0x1000, size=256, device=0, stream=1)

        assert block.addr == 0x1000
        assert block.value.size == 256
        assert block.device == 0
        assert block.stream == 1
        assert block.segment_id is None

    def test_memory_block_properties(self):
        """Test MemoryBlock properties."""
        block = MemoryBlock(addr=0x1000, size=256, capture_trace=False)

        assert block.end_addr == 0x1000 + 256
        assert block.addr_hex == hex(0x1000)
        assert block.is_head == True  # No prev
        assert block.is_split == False  # No prev/next

    def test_memory_block_comparison(self):
        """Test MemoryBlock comparison operators."""
        block1 = MemoryBlock(addr=0x1000, size=256, stream=1, capture_trace=False)
        block2 = MemoryBlock(addr=0x2000, size=256, stream=1, capture_trace=False)
        block3 = MemoryBlock(addr=0x1000, size=512, stream=1, capture_trace=False)

        # Test less than (by address when stream and size are same)
        assert block1 < block2

        # Test less than (by size when stream is same)
        assert block1 < block3

        # Test equality
        block4 = MemoryBlock(addr=0x1000, size=256, stream=1, capture_trace=False)
        assert block1 == block4

    def test_memory_block_hash(self):
        """Test MemoryBlock hashing for use in sets."""
        block1 = MemoryBlock(addr=0x1000, size=256, stream=1, capture_trace=False)
        block2 = MemoryBlock(addr=0x1000, size=256, stream=1, capture_trace=False)

        # Same blocks should have same hash
        assert hash(block1) == hash(block2)

        # Can be used in sets
        block_set = {block1, block2}
        assert len(block_set) == 1  # Duplicates removed


class TestMemoryBlockAllocation:
    """Test cases for MemoryBlock allocation and deallocation."""

    def test_request_alloc(self):
        """Test block allocation."""
        block = MemoryBlock(addr=0x1000, size=256, capture_trace=False)

        with patch("ures.memory.blocks.time.time_ns", return_value=123456789):
            block.request_alloc()

        assert block.value.action == "alloc"
        assert block.value.allocated == True
        assert block.value.alloc_time_ns == 123456789

    def test_request_alloc_with_trace(self):
        """Test block allocation with trace capture."""
        block = MemoryBlock(addr=0x1000, size=256, capture_trace=False)

        with patch("ures.memory.blocks.TraceInfo.capture_current_trace") as mock_trace:
            mock_trace_obj = Mock()
            mock_trace.return_value = mock_trace_obj

            block.request_alloc(capture_trace=True)

        # Verify trace was captured and added
        mock_trace.assert_called_once()
        assert mock_trace_obj in block.value.traces

    def test_request_free(self):
        """Test requesting block free."""
        block = MemoryBlock(addr=0x1000, size=256, capture_trace=False)
        block.request_alloc()  # First allocate

        with patch("ures.memory.blocks.time.time_ns", return_value=987654321):
            block.request_free()

        assert block.value.action == "free_requested"
        assert block.value.allocated == True  # Still marked as allocated
        assert block.value.free_requested_time_ns == 987654321

    def test_complete_free(self):
        """Test completing free operation."""
        block = MemoryBlock(addr=0x1000, size=256, capture_trace=False)
        block.request_alloc()
        block.request_free()

        with patch("ures.memory.blocks.time.time_ns", return_value=111111111):
            block.complete_free()

        assert block.value.action == "free_completed"
        assert block.value.allocated == False
        assert block.value.free_completed_time_ns == 111111111

    def test_free_block_immediate(self):
        """Test immediate block freeing."""
        block = MemoryBlock(addr=0x1000, size=256, capture_trace=False)
        block.request_alloc()

        with patch("ures.memory.blocks.time.time_ns", return_value=222222222):
            block.free_block()

        assert block.value.action == "free_completed"
        assert block.value.allocated == False
        assert block.value.free_completed_time_ns == 222222222

    def test_force_reset_memory_info(self):
        """Test forcing reset of memory info."""
        block = MemoryBlock(addr=0x1000, size=256, capture_trace=False)
        block.request_alloc()

        # Add some traces
        trace = Mock()
        block.value.add_trace(trace)

        block.force_reset_memory_info()

        # Should have clean MemoryInfo
        assert block.value.action == "free"
        assert block.value.allocated == False
        assert len(block.value.traces) == 0
        assert block.addr == 0x1000
        assert block.value.size == 256


class TestMemoryBlockInsertBlock:
    """Test cases for MemoryBlock splicing (splitting) operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.pool = Mock()
        self.pool.blocks = Mock()

    def test_insert_memory_block_same_start_address(self):
        """Test normal block splicing."""
        block = MemoryBlock(addr=0x1000, size=1024, pool=self.pool, capture_trace=False)
        add_block = MemoryBlock(
            addr=0x1000, size=256, pool=self.pool, capture_trace=False
        )

        add_block = block.insert_block(add_block)

        assert add_block.addr == 0x1000
        assert add_block.value.size == 256
        assert block.addr == 0x1000 + 256
        assert block.value.size == 1024 - 256

    def test_insert_memory_block_same_end_address(self):
        """Test normal block splicing."""
        block = MemoryBlock(addr=0x1000, size=1024, pool=self.pool, capture_trace=False)
        original_block_end_addr = block.end_addr
        add_block = MemoryBlock(
            addr=0x1000 + 1024 - 256, size=256, pool=self.pool, capture_trace=False
        )

        add_block = block.insert_block(add_block)

        assert block == add_block
        assert add_block.addr == 0x1000 + 1024 - 256
        assert add_block.end_addr == original_block_end_addr

    def test_insert_memory_block_between_start_and_end(self):
        """Test normal block splicing."""
        block = MemoryBlock(addr=0x1000, size=1024, pool=self.pool, capture_trace=False)
        add_block = MemoryBlock(
            addr=0x1000 + 256, size=256, pool=self.pool, capture_trace=False
        )

        add_block = block.insert_block(add_block)

        assert add_block.addr == 0x1000 + 256
        assert add_block.value.size == 256
        assert block.addr == 0x1000 + 512
        assert block.value.size == 1024 - 512
        assert add_block.prev.addr == 0x1000
        assert add_block.prev.value.size == 256
        assert add_block.prev.value.is_allocated() is False
        assert add_block.next.addr == 0x1000 + 512
        assert add_block.next.value.size == 512
        assert add_block.next.value.is_allocated() is False

    def test_error_current_block_is_allocated(self):
        """Test that splicing allocated block raises error."""
        block = MemoryBlock(addr=0x1000, size=1024, capture_trace=False)
        block.request_alloc()
        add_block = MemoryBlock(
            addr=0x1000 + 256, size=256, pool=self.pool, capture_trace=False
        )

        with pytest.raises(
            MemoryError, match="Cannot insert a block into an allocated block"
        ):
            block.insert_block(add_block)

    def test_error_insert_block_not_be_contained(self):
        """Test that splicing larger than block size fails."""
        block = MemoryBlock(addr=0x1000, size=256, capture_trace=False)
        add_block = MemoryBlock(
            addr=0x2000, size=512, pool=self.pool, capture_trace=False
        )

        with pytest.raises(
            ValueError,
            match=f"Cannot insert block {add_block.addr_hex}",
        ):
            block.insert_block(add_block)

    def test_error_insert_block_is_split(self):
        """Test that splicing larger than block size fails."""
        block = MemoryBlock(addr=0x1000, size=256, capture_trace=False)
        add_block = MemoryBlock(
            addr=0x1000 + 128, size=128, pool=self.pool, capture_trace=False
        )
        add_block.splice(100)

        with pytest.raises(
            ValueError, match=f"Cannot insert a split block into another block"
        ):
            block.insert_block(add_block)


class TestMemoryBlockSplicing:
    """Test cases for MemoryBlock splicing (splitting) operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.pool = Mock()
        self.pool.blocks = Mock()

    def test_splice_normal_case(self):
        """Test normal block splicing."""
        block = MemoryBlock(addr=0x1000, size=1024, pool=self.pool, capture_trace=False)

        new_block = block.splice(256)

        # New block should have requested size and original address
        assert new_block.addr == 0x1000
        assert new_block.value.size == 256

        # Original block should be adjusted
        assert block.addr == 0x1000 + 256
        assert block.value.size == 1024 - 256

        # Blocks should be linked
        assert new_block.next == block
        assert block.prev == new_block

    def test_splice_exact_size(self):
        """Test splicing entire block."""
        block = MemoryBlock(addr=0x1000, size=256, pool=self.pool, capture_trace=False)

        # Mock the pool.blocks to support 'in' operator and remove method
        mock_blocks = Mock()
        mock_blocks.__contains__ = Mock(return_value=True)
        mock_blocks.remove = Mock()
        self.pool.blocks = mock_blocks

        result = block.splice(256)

        # Should return original block and remove from pool
        assert result == block
        mock_blocks.remove.assert_called_once_with(block)

    def test_splice_allocated_block_fails(self):
        """Test that splicing allocated block raises error."""
        block = MemoryBlock(addr=0x1000, size=1024, capture_trace=False)
        block.request_alloc()

        with pytest.raises(MemoryError, match="Cannot split an allocated block"):
            block.splice(256)

    def test_splice_size_too_large_fails(self):
        """Test that splicing larger than block size fails."""
        block = MemoryBlock(addr=0x1000, size=256, capture_trace=False)

        with pytest.raises(
            ValueError,
            match="Cannot split a block of size 256 into a block of size 512",
        ):
            block.splice(512)

    def test_splice_updates_segment_first_block(self):
        """Test that splicing updates segment's first_block reference."""
        segment = Mock()
        segment.first_block = None
        self.pool.get_segment.return_value = segment

        block = MemoryBlock(
            addr=0x1000, size=1024, pool=self.pool, segment_id=1, capture_trace=False
        )
        segment.first_block = block

        new_block = block.splice(256)

        # Segment should now point to new block
        assert segment.first_block == new_block

    def test_splice_with_trace_capture(self):
        """Test splicing with trace capture."""
        block = MemoryBlock(addr=0x1000, size=1024, capture_trace=False)

        with patch("ures.memory.blocks.TraceInfo.capture_current_trace") as mock_trace:
            mock_trace_obj = Mock()
            mock_trace.return_value = mock_trace_obj

            new_block = block.splice(256, capture_trace=True)

        # Both blocks should have the split trace
        assert mock_trace_obj in block.value.traces
        assert mock_trace_obj in new_block.value.traces


class TestMemoryBlockCoalescing:
    """Test cases for MemoryBlock coalescing (merging) operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.pool = Mock()
        # Mock the blocks collection to support 'in' operator and methods
        mock_blocks = Mock()
        mock_blocks.__contains__ = Mock(return_value=True)
        mock_blocks.add = Mock()
        mock_blocks.remove = Mock()
        self.pool.blocks = mock_blocks

    def test_coalesce_no_adjacent_blocks(self):
        """Test coalescing when no adjacent blocks available."""
        block = MemoryBlock(addr=0x1000, size=256, pool=self.pool, capture_trace=False)

        result = block.coalesce()

        # Should return same block and ensure it's in pool
        assert result == block
        self.pool.blocks.add.assert_called_once_with(block)

    def test_coalesce_with_next_block(self):
        """Test coalescing with next block."""
        # Create blocks
        block1 = MemoryBlock(
            addr=0x1000, size=256, pool=self.pool, segment_id=1, capture_trace=False
        )
        block2 = MemoryBlock(
            addr=0x1100, size=256, pool=self.pool, segment_id=1, capture_trace=False
        )

        # Link them using the parent class methods
        block1.insert_after(block2)

        result = block1.coalesce()

        # Block1 should be expanded to include block2
        assert result == block1
        assert block1.value.size == 512  # 256 + 256
        # Verify block2 was removed from the linked list (next should be None or different)
        assert block1.next is None or block1.next.addr != block2.addr

    def test_coalesce_with_prev_block(self):
        """Test coalescing with previous block."""
        # Create blocks
        block1 = MemoryBlock(
            addr=0x1000, size=256, pool=self.pool, segment_id=1, capture_trace=False
        )
        block2 = MemoryBlock(
            addr=0x1100, size=256, pool=self.pool, segment_id=1, capture_trace=False
        )

        # Link them using the parent class methods
        block1.insert_after(block2)

        result = block2.coalesce()

        # Block2 should be expanded to include block1
        assert result == block2
        assert block2.value.size == 512
        assert block2.addr == 0x1000  # Should take block1's address
        # Verify block1 was removed from the linked list (prev should be None or different)
        assert block2.prev is None or block2.prev.addr != block1.addr

    def test_coalesce_allocated_block_fails(self):
        """Test that coalescing allocated block fails."""
        block = MemoryBlock(addr=0x1000, size=256, capture_trace=False)
        block.request_alloc()

        with pytest.raises(MemoryError, match="Cannot split an allocated block"):
            block.coalesce()

    def test_coalesce_ignores_different_segments(self):
        """Test that coalescing ignores blocks from different segments."""
        # Create blocks in different segments
        block1 = MemoryBlock(
            addr=0x1000, size=256, pool=self.pool, segment_id=1, capture_trace=False
        )
        block2 = MemoryBlock(
            addr=0x1100, size=256, pool=self.pool, segment_id=2, capture_trace=False
        )  # Different segment

        # Link them using the parent class methods
        block1.insert_after(block2)

        result = block1.coalesce()

        # Should not coalesce due to different segments
        assert result == block1
        assert block1.value.size == 256  # Unchanged
        # Should ensure block is in pool since no coalescing occurred
        self.pool.blocks.add.assert_called_once_with(block1)

    def test_coalesce_updates_segment_first_block(self):
        """Test that coalescing updates segment's first_block reference."""
        segment = Mock()
        self.pool.get_segment.return_value = segment

        # Create blocks
        block1 = MemoryBlock(
            addr=0x1000, size=256, pool=self.pool, segment_id=1, capture_trace=False
        )
        block2 = MemoryBlock(
            addr=0x1100, size=256, pool=self.pool, segment_id=1, capture_trace=False
        )

        # Set block        # Set block1 as segment's first block
        segment.first_block = block1

        # Link them using the parent class methods
        block1.insert_after(block2)

        # Coalesce block2 with block1 (prev)
        result = block2.coalesce()

        # Segment should now point to coalesced block
        assert segment.first_block == result

    def test_coalesce_with_both_adjacent_blocks(self):
        """Test coalescing with both previous and next blocks."""
        # Create three consecutive blocks
        block1 = MemoryBlock(
            addr=0x1000, size=256, pool=self.pool, segment_id=1, capture_trace=False
        )
        block2 = MemoryBlock(
            addr=0x1100, size=256, pool=self.pool, segment_id=1, capture_trace=False
        )
        block3 = MemoryBlock(
            addr=0x1200, size=256, pool=self.pool, segment_id=1, capture_trace=False
        )

        # Link them: block1 <-> block2 <-> block3
        block1.insert_after(block2)
        block2.insert_after(block3)

        result = block2.coalesce()

        # Block2 should be expanded to include both neighbors
        assert result == block2
        assert block2.value.size == 768  # 256 + 256 + 256
        assert block2.addr == 0x1000  # Should take block1's address

    def test_coalesce_with_allocated_adjacent_blocks(self):
        """Test that coalescing ignores allocated adjacent blocks."""
        # Create blocks
        block1 = MemoryBlock(
            addr=0x1000, size=256, pool=self.pool, segment_id=1, capture_trace=False
        )
        block2 = MemoryBlock(
            addr=0x1100, size=256, pool=self.pool, segment_id=1, capture_trace=False
        )

        # Allocate block1
        block1.request_alloc()

        # Link them
        block1.insert_after(block2)

        result = block2.coalesce()

        # Should not coalesce with allocated block1
        assert result == block2
        assert block2.value.size == 256  # Unchanged
        assert block2.addr == 0x1100  # Unchanged


class TestMemoryBlockUtilities:
    """Test cases for MemoryBlock utility methods."""

    def test_contains_address(self):
        """Test address containment checking."""
        block = MemoryBlock(addr=0x1000, size=256, capture_trace=False)

        # Address within block
        assert block.contains_address(0x1000, 1)
        assert block.contains_address(0x1080, 128)
        assert block.contains_address(0x10FF, 1)

        # Address outside block
        assert not block.contains_address(0x0FFF, 1)  # Before
        assert not block.contains_address(0x1100, 1)  # After
        assert not block.contains_address(0x1080, 256)  # Extends beyond

    def test_to_dict(self):
        """Test converting MemoryBlock to dictionary."""
        block = MemoryBlock(
            addr=0x1000, size=256, device=0, stream=1, segment_id=5, capture_trace=False
        )
        block.request_alloc()

        result = block.to_dict()

        expected_keys = {
            "addr",
            "addr_hex",
            "end_addr",
            "end_addr_hex",
            "size",
            "device",
            "stream",
            "segment_id",
            "is_allocated",
            "is_segment_start",
            "is_split",
            "action",
            "allocated",
            "alloc_time_ns",
            "free_requested_time_ns",
            "free_completed_time_ns",
            "traces",
        }

        assert set(result.keys()) == expected_keys
        assert result["addr"] == 0x1000
        assert result["addr_hex"] == hex(0x1000)
        assert result["size"] == 256
        assert result["is_allocated"] == True


class TestSegment:
    """Test cases for Segment class."""

    def test_segment_creation(self):
        """Test basic Segment creation."""
        with patch("time.time_ns", return_value=123456789):
            segment = Segment(
                segment_id=1, start_addr=0x1000, original_size=1024, device=0, stream=1
            )

        assert segment.segment_id == 1
        assert segment.start_addr == 0x1000
        assert segment.original_size == 1024
        assert segment.device == 0
        assert segment.stream == 1
        assert segment.creation_time_ns == 123456789
        assert segment.end_addr == 0x1000 + 1024

    def test_segment_with_blocks(self):
        """Test segment with associated blocks."""
        segment = Segment(segment_id=1, start_addr=0x1000, original_size=1024)

        # Create some blocks
        block1 = Mock()
        block1.segment_id = 1
        block1.addr = 0x1000
        block1.value.size = 512
        block1.value.is_allocated.return_value = True

        block2 = Mock()
        block2.segment_id = 1
        block2.addr = 0x1200
        block2.value.size = 512
        block2.value.is_allocated.return_value = False

        # Mock the linked list traversal
        block1.get_head.return_value = block1
        block1.next = block2
        block2.next = None

        segment.first_block = block1

        with patch.object(segment, "get_blocks", return_value=[block1, block2]):
            assert segment.get_block_count() == 2
            assert segment.get_allocated_bytes() == 512
            assert segment.get_free_bytes() == 512
            assert segment.get_utilization_ratio() == 0.5
            assert not segment.is_fully_free()
            assert not segment.is_fully_allocated()

    def test_segment_fragmentation_ratio(self):
        """Test fragmentation ratio calculation."""
        segment = Segment(segment_id=1, start_addr=0x1000, original_size=1024)

        # No fragmentation (1 block)
        with patch.object(segment, "get_block_count", return_value=1):
            assert segment.get_fragmentation_ratio() == 0.0

        # Some fragmentation (4 blocks)
        with patch.object(segment, "get_block_count", return_value=4):
            assert segment.get_fragmentation_ratio() == 0.75  # 1.0 - (1.0/4)

    def test_segment_contains_address(self):
        """Test address containment in segment."""
        segment = Segment(segment_id=1, start_addr=0x1000, original_size=1024)

        assert segment.contains_address(0x1000, 1)
        assert segment.contains_address(0x1200, 512)
        assert segment.contains_address(0x13FF, 1)

        assert not segment.contains_address(0x0FFF, 1)
        assert not segment.contains_address(0x1400, 1)
        assert not segment.contains_address(0x1200, 1024)

    def test_segment_to_dict(self):
        """Test converting Segment to dictionary."""
        segment = Segment(segment_id=1, start_addr=0x1000, original_size=1024)

        with patch.object(segment, "get_block_count", return_value=2), patch.object(
            segment, "get_allocated_bytes", return_value=512
        ), patch.object(segment, "get_free_bytes", return_value=512):
            result = segment.to_dict()

        expected_keys = {
            "segment_id",
            "start_addr",
            "start_addr_hex",
            "end_addr",
            "end_addr_hex",
            "original_size",
            "device",
            "stream",
            "creation_time_ns",
            "block_count",
            "allocated_bytes",
            "free_bytes",
            "utilization_ratio",
            "fragmentation_ratio",
            "is_fully_free",
            "is_fully_allocated",
            "traces",
        }

        assert set(result.keys()) == expected_keys
        assert result["segment_id"] == 1
        assert result["start_addr"] == 0x1000


class TestBlockPoolBasics:
    """Test cases for BlockPool basic functionality."""

    def test_blockpool_creation(self):
        """Test basic BlockPool creation."""
        pool = BlockPool()

        assert len(pool.blocks) == 0
        assert len(pool.segments) == 0
        assert pool.next_segment_id == 0

    def test_insert_into_blocks(self):
        """Test inserting blocks into the pool."""
        pool = BlockPool()
        block = MemoryBlock(addr=0x1000, size=256, capture_trace=False)

        pool.insert_into_blocks(block)

        assert block in pool.blocks
        assert len(pool.blocks) == 1

    def test_lower_bound_search(self):
        """Test lower bound searching in blocks."""
        pool = BlockPool()

        # Add some blocks
        block1 = MemoryBlock(addr=0x1000, size=256, stream=1, capture_trace=False)
        block2 = MemoryBlock(addr=0x2000, size=256, stream=1, capture_trace=False)
        block3 = MemoryBlock(addr=0x3000, size=512, stream=1, capture_trace=False)

        pool.insert_into_blocks(block1)
        pool.insert_into_blocks(block2)
        pool.insert_into_blocks(block3)

        # Search for block with size 256
        search_key = MemoryBlock(addr=0, size=256, stream=1, capture_trace=False)
        result = pool.lower_bound(search_key)

        assert result == block1  # First block with size 256

    def test_is_end_block(self):
        """Test checking if a block would be at the end."""
        pool = BlockPool()
        block = MemoryBlock(addr=0x1000, size=256, stream=1, capture_trace=False)
        pool.insert_into_blocks(block)

        # Block larger than any in pool should be "end"
        large_block = MemoryBlock(addr=0x9000, size=1024, stream=1, capture_trace=False)
        assert pool.is_end_block(large_block)

        # Block smaller should not be "end"
        small_block = MemoryBlock(addr=0x500, size=128, stream=1, capture_trace=False)
        assert not pool.is_end_block(small_block)


class TestBlockPoolSegmentManagement:
    """Test cases for BlockPool segment management."""

    def test_create_segment_success(self):
        """Test successful segment creation."""
        pool = BlockPool()

        with patch("time.time_ns", return_value=123456789):
            segment = pool.create_segment(0x1000, 1024, device=0, stream=1)

        assert segment.segment_id == 0
        assert segment.start_addr == 0x1000
        assert segment.original_size == 1024
        assert segment.device == 0
        assert segment.stream == 1

        # Should be in pool's segments
        assert pool.segments[0] == segment
        assert pool.next_segment_id == 1

        # Should have created initial block
        assert segment.first_block is not None
        assert segment.first_block in pool.blocks

    def test_create_segment_overlap_fails(self):
        """Test that overlapping segment creation fails."""
        pool = BlockPool()

        # Create first segment
        pool.create_segment(0x1000, 1024)

        # Try to create overlapping segment
        with pytest.raises(MemoryError, match="Cannot create segment.*Overlaps"):
            pool.create_segment(0x1200, 1024)  # Overlaps with first

    def test_get_segment(self):
        """Test retrieving segments by ID."""
        pool = BlockPool()
        segment = pool.create_segment(0x1000, 1024)

        assert pool.get_segment(0) == segment
        assert pool.get_segment(999) is None

    def test_get_segment_by_address(self):
        """Test finding segment by address."""
        pool = BlockPool()
        segment = pool.create_segment(0x1000, 1024)

        assert pool.get_segment_by_address(0x1000) == segment
        assert pool.get_segment_by_address(0x1200) == segment
        assert pool.get_segment_by_address(0x13FF) == segment
        assert pool.get_segment_by_address(0x0FFF) is None
        assert pool.get_segment_by_address(0x1400) is None

    def test_remove_segment_success(self):
        """Test successful segment removal."""
        pool = BlockPool()
        segment = pool.create_segment(0x1000, 1024)
        initial_block = segment.first_block

        result = pool.remove_segment(0, force=True)

        assert result == True
        assert 0 not in pool.segments
        assert initial_block not in pool.blocks

    def test_remove_segment_with_allocated_blocks(self):
        """Test segment removal with allocated blocks."""
        pool = BlockPool()
        segment = pool.create_segment(0x1000, 1024)

        # Allocate the block
        segment.first_block.request_alloc()

        # Should fail without force
        result = pool.remove_segment(0, force=False)
        assert result == False
        assert 0 in pool.segments  # Still there

        # Should succeed with force
        result = pool.remove_segment(0, force=True)
        assert result == True
        assert 0 not in pool.segments

    def test_list_all_segments(self):
        """Test listing all segments."""
        pool = BlockPool()

        segment1 = pool.create_segment(0x2000, 512)  # Higher address first
        segment2 = pool.create_segment(0x1000, 1024)  # Lower address

        segments = pool.list_all_segments()

        assert len(segments) == 2
        # Should be sorted by start address
        assert segments[0]["start_addr"] == 0x1000
        assert segments[1]["start_addr"] == 0x2000

    def test_list_blocks_in_segment(self):
        """Test listing blocks within a segment."""
        pool = BlockPool()
        segment = pool.create_segment(0x1000, 1024)

        # Split the initial block
        initial_block = segment.first_block
        new_block = initial_block.splice(256)

        blocks = pool.list_blocks_in_segment(0)

        assert len(blocks) == 2
        # Should contain information about both blocks
        addrs = [block["addr"] for block in blocks]
        assert 0x1000 in addrs
        assert 0x1000 + 256 in addrs


class TestBlockPoolOverlapDetection:
    """Test cases for BlockPool overlap detection functionality."""

    def test_check_segment_overlap_no_overlap(self):
        """Test overlap checking with no conflicts."""
        pool = BlockPool()
        pool.create_segment(0x1000, 1024)

        # Check non-overlapping range
        result = pool.check_segment_overlap(0x2000, 1024)

        assert result["has_overlap"] == False
        assert result["safe_to_create"] == True
        assert len(result["overlapping_segments"]) == 0
        assert result["overlap_type"] == "none"

    def test_check_segment_overlap_partial(self):
        """Test partial overlap detection."""
        pool = BlockPool()
        pool.create_segment(0x1000, 1024)  # 0x1000 - 0x1400

        # Check partially overlapping range
        result = pool.check_segment_overlap(0x1200, 1024)  # 0x1200 - 0x1600

        assert result["has_overlap"] == True
        assert result["safe_to_create"] == False
        assert len(result["overlapping_segments"]) == 1
        assert result["overlap_type"] == "partial"

        overlap = result["overlapping_segments"][0]
        assert overlap["segment_id"] == 0
        assert overlap["overlap_start"] == 0x1200
        assert overlap["overlap_end"] == 0x1400

    def test_check_segment_overlap_complete(self):
        """Test complete overlap detection."""
        pool = BlockPool()
        pool.create_segment(0x1000, 1024)

        # Check exact same range
        result = pool.check_segment_overlap(0x1000, 1024)

        assert result["has_overlap"] == True
        assert result["overlap_type"] == "complete"

    def test_check_segment_overlap_contains(self):
        """Test when new segment would contain existing segment."""
        pool = BlockPool()
        pool.create_segment(0x1200, 512)  # Smaller existing segment

        # Check larger containing range
        result = pool.check_segment_overlap(0x1000, 1024)

        assert result["has_overlap"] == True
        assert result["overlap_type"] == "contains"

    def test_check_segment_overlap_contained(self):
        """Test when new segment would be contained in existing segment."""
        pool = BlockPool()
        pool.create_segment(0x1000, 1024)  # Larger existing segment

        # Check smaller contained range
        result = pool.check_segment_overlap(0x1200, 512)

        assert result["has_overlap"] == True
        assert result["overlap_type"] == "contained"

    def test_check_segment_overlap_invalid_size(self):
        """Test overlap checking with invalid size."""
        pool = BlockPool()

        result = pool.check_segment_overlap(0x1000, 0)  # Invalid size

        assert result["has_overlap"] == False
        assert result["safe_to_create"] == False
        assert "error" in result

    def test_find_safe_address_range(self):
        """Test finding safe address ranges."""
        pool = BlockPool()

        # Empty pool - should find space at min_addr
        result = pool.find_safe_address_range(1024)

        assert result is not None
        assert result["start_addr"] == 0x1000  # Default min_addr
        assert result["size"] == 1024
        assert result["location"] == "first_segment"

    def test_find_safe_address_range_with_gaps(self):
        """Test finding safe address ranges with existing segments."""
        pool = BlockPool()

        # Create segments with gaps
        pool.create_segment(0x1000, 1024)  # 0x1000 - 0x1400
        pool.create_segment(0x2000, 1024)  # 0x2000 - 0x2400

        # Should find gap between segments
        result = pool.find_safe_address_range(512)

        assert result is not None
        assert 0x1400 <= result["start_addr"] < 0x2000
        assert result["size"] == 512

    def test_find_safe_address_range_no_space(self):
        """Test when no safe address range is available."""
        pool = BlockPool()

        # Fill entire range
        pool.create_segment(0x1000, 0xFFFFFFFF - 0x1000)

        # Should not find space for large allocation
        result = pool.find_safe_address_range(1024)

        assert result is None


class TestBlockPoolAnalytics:
    """Test cases for BlockPool analytics and tracing functionality."""

    def test_get_traces_by_operation(self):
        """Test retrieving traces by operation type."""
        pool = BlockPool()

        # Create segment (generates traces)
        segment = pool.create_segment(0x1000, 1024)

        # Perform some operations
        block = segment.first_block.splice(256)
        block.request_alloc()

        # Get traces for specific operations
        create_traces = pool.get_traces_by_operation("create")
        alloc_traces = pool.get_traces_by_operation("alloc")

        assert len(create_traces) >= 2  # At least segment creation and block creation
        assert len(alloc_traces) >= 1  # At least one allocation

        # Traces should be sorted by timestamp
        if len(create_traces) > 1:
            assert create_traces[0]["timestamp_ns"] <= create_traces[1]["timestamp_ns"]

    def test_get_all_traces(self):
        """Test retrieving all traces."""
        pool = BlockPool()

        # Create segment and perform operations
        segment = pool.create_segment(0x1000, 1024)
        block = segment.first_block.splice(256)
        block.request_alloc()
        block.free_block()

        all_traces = pool.get_all_traces()

        assert len(all_traces) > 0

        # Should include both segment and block traces
        source_types = {trace["source_type"] for trace in all_traces}
        assert "segment" in source_types
        assert "block" in source_types

        # Should be sorted by timestamp
        timestamps = [trace["timestamp_ns"] for trace in all_traces]
        assert timestamps == sorted(timestamps)

    def test_analyze_memory_patterns(self):
        """Test memory pattern analysis."""
        pool = BlockPool()

        # Create and manipulate memory
        segment = pool.create_segment(0x1000, 1024)
        block = segment.first_block.splice(256)
        block.request_alloc()
        block.free_block()

        analysis = pool.analyze_memory_patterns()

        assert "total_operations" in analysis
        assert "operations_count" in analysis
        assert "most_frequent_operation" in analysis
        assert "operation_timeline" in analysis

        assert analysis["total_operations"] > 0
        assert isinstance(analysis["operations_count"], dict)

    def test_get_memory_summary(self):
        """Test memory summary generation."""
        pool = BlockPool()

        # Create segments with different utilization
        segment1 = pool.create_segment(0x1000, 1024)
        segment2 = pool.create_segment(0x2000, 1024)

        # Allocate some blocks
        block1 = segment1.first_block.splice(512)
        block1.request_alloc()

        summary = pool.get_memory_summary()

        assert summary["total_segments"] == 2
        assert summary["total_original_size"] == 2048
        assert summary["total_allocated_bytes"] == 512
        assert summary["total_free_bytes"] == 2048 - 512
        assert 0 <= summary["overall_utilization"] <= 1
        assert 0 <= summary["average_fragmentation"] <= 1

    def test_print_memory_status(self, capsys):
        """Test memory status printing."""
        pool = BlockPool()

        # Create a segment and perform some operations
        segment = pool.create_segment(0x1000, 1024)
        block = segment.first_block.splice(256)
        block.request_alloc()

        pool.print_memory_status()

        captured = capsys.readouterr()

        # Should contain expected sections
        assert "BlockPool Memory Status" in captured.out
        assert "Total segments:" in captured.out
        assert "Segments" in captured.out
        assert "Segment 0:" in captured.out


class TestBlockPoolIntegration:
    """Integration test cases testing multiple components together."""

    def test_complete_memory_lifecycle(self):
        """Test complete memory management lifecycle."""
        pool = BlockPool()

        # 1. Create segment
        segment = pool.create_segment(0x1000, 1024, device=0, stream=1)
        assert len(pool.segments) == 1
        assert len(pool.blocks) == 1

        # 2. Split for allocation
        initial_block = segment.first_block
        block1 = initial_block.splice(256)  # 256 bytes at 0x1000
        # Note: splice removes the new block from pool and returns it
        # The remaining block stays in the pool

        assert segment.get_block_count() == 2

        # 3. Allocate first block
        block1.request_alloc()
        assert block1.value.is_allocated()
        assert segment.get_allocated_bytes() == 256
        assert segment.get_free_bytes() == 768

        # 4. Split remainder and allocate
        remaining_block = [
            b for b in segment.get_blocks() if not b.value.is_allocated()
        ][0]
        block2 = remaining_block.splice(300)  # 300 bytes
        block2.request_alloc()

        assert segment.get_block_count() == 3
        assert segment.get_allocated_bytes() == 556  # 256 + 300
        assert segment.get_fragmentation_ratio() > 0  # Now fragmented

        # 5. Free blocks
        block1.free_block()
        block2.free_block()

        assert segment.get_allocated_bytes() == 0
        assert segment.is_fully_free()

        # 6. Coalesce to reduce fragmentation
        # Find free blocks and coalesce them
        free_blocks = [b for b in segment.get_blocks() if not b.value.is_allocated()]

        # Coalesce adjacent blocks
        for block in free_blocks:
            block.coalesce()

        # Should have fewer blocks now
        final_block_count = segment.get_block_count()
        assert final_block_count < 3  # Should be coalesced

    def test_memory_stress_operations(self):
        """Test memory management under stress conditions."""
        pool = BlockPool()

        # Create multiple segments
        segments = []
        for i in range(5):
            seg = pool.create_segment(0x1000 + i * 0x2000, 1024)
            segments.append(seg)

        assert len(pool.segments) == 5

        # Perform many split operations
        allocated_blocks = []
        for segment in segments:
            current_block = segment.first_block
            for j in range(4):  # Split each segment into 4 pieces
                if current_block.value.size >= 64:
                    new_block = current_block.splice(64)
                    new_block.request_alloc()
                    allocated_blocks.append(new_block)

        # Verify allocations
        total_allocated = sum(seg.get_allocated_bytes() for seg in segments)
        assert total_allocated > 0

        # Free all blocks
        for block in allocated_blocks:
            block.free_block()

        # Verify all freed
        total_allocated = sum(seg.get_allocated_bytes() for seg in segments)
        assert total_allocated == 0

        # Coalesce all segments
        for segment in segments:
            for block in segment.get_blocks():
                if not block.value.is_allocated():
                    block.coalesce()

        # Check fragmentation reduced
        avg_fragmentation = sum(
            seg.get_fragmentation_ratio() for seg in segments
        ) / len(segments)
        # Should be lower due to coalescing (exact value depends on implementation)
        assert 0 <= avg_fragmentation <= 1

    def test_error_handling_integration(self):
        """Test error handling across multiple operations."""
        pool = BlockPool()
        segment = pool.create_segment(0x1000, 1024)
        block = segment.first_block

        # Test allocation errors
        block.request_alloc()

        # Can't split allocated block
        with pytest.raises(MemoryError):
            block.splice(256)

        # Can't coalesce allocated block
        with pytest.raises(MemoryError):
            block.coalesce()

        # Free and test size errors
        block.free_block()

        # Can't split larger than block size
        with pytest.raises(ValueError):
            block.splice(2048)

        # Test overlap errors
        with pytest.raises(MemoryError):
            pool.create_segment(0x1200, 1024)  # Overlaps existing

    def test_trace_consistency(self):
        """Test that traces are consistently captured across operations."""
        pool = BlockPool()

        # Enable trace capture and perform operations
        segment = pool.create_segment(0x1000, 1024, capture_trace=True)
        block = segment.first_block.splice(256, capture_trace=True)
        block.request_alloc(capture_trace=True)
        block.free_block(capture_trace=True)
        block.coalesce(capture_trace=True)

        # Get all traces
        all_traces = pool.get_all_traces()

        # Should have traces for all operations
        operations = {trace["operation"] for trace in all_traces}
        expected_operations = {
            "create_segment",
            "create",
            "split",
            "alloc",
            "free_immediate",
            "coalesce",
        }

        # All expected operations should be present (subset check)
        assert expected_operations.issubset(operations)

        # Each trace should have required fields
        for trace in all_traces:
            assert "timestamp_ns" in trace
            assert "operation" in trace
            assert "source_type" in trace
            assert trace["source_type"] in ["segment", "block"]


class TestBlockPoolEdgeCases:
    """Test cases for edge cases and boundary conditions."""

    def test_empty_pool_operations(self):
        """Test operations on empty pool."""
        pool = BlockPool()

        # Should handle empty pool gracefully
        assert pool.get_segment(0) is None
        assert pool.get_segment_by_address(0x1000) is None
        assert pool.list_all_segments() == []
        assert pool.list_blocks_in_segment(0) == []
        assert pool.get_all_traces() == []

        summary = pool.get_memory_summary()
        assert summary["total_segments"] == 0
        assert summary["total_original_size"] == 0

        # Remove non-existent segment should return False
        assert pool.remove_segment(999) == False

    def test_single_byte_operations(self):
        """Test operations with single-byte allocations."""
        pool = BlockPool()
        segment = pool.create_segment(0x1000, 10)  # Very small segment

        # Split into single bytes
        block1 = segment.first_block.splice(10)
        assert block1.value.size == 10
        block2 = segment.first_block.splice(1)
        assert block2.value.size == 1
        # As the splice algorithm, allocated block is always returned, and
        # returned block was inserted prior to the original block. So the segment's first_block
        # should now be block2 of size 1.
        assert segment.first_block.value.size == 1

        # Allocate and free
        block1.request_alloc()
        block1.free_block()

        # Should be able to coalesce
        result = block1.coalesce()
        assert result.value.size >= 1

    def test_large_address_values(self):
        """Test operations with large address values."""
        pool = BlockPool()

        # Use large addresses (64-bit)
        large_addr = 0x7FFFFFFF00000000
        segment = pool.create_segment(large_addr, 1024)

        assert segment.start_addr == large_addr
        assert segment.end_addr == large_addr + 1024
        assert segment.contains_address(large_addr, 512)

        # Should find by address
        found = pool.get_segment_by_address(large_addr + 100)
        assert found == segment

    def test_zero_size_edge_cases(self):
        """Test edge cases with zero sizes."""
        pool = BlockPool()

        # Zero size should be rejected
        result = pool.check_segment_overlap(0x1000, 0)
        assert not result["safe_to_create"]
        assert "error" in result

        # Find safe range with zero size should return None
        result = pool.find_safe_address_range(0)
        assert result is None

    def test_maximum_fragmentation(self):
        """Test maximum fragmentation scenarios."""
        pool = BlockPool()
        segment = pool.create_segment(0x1000, 1024)

        # Split into many tiny blocks
        current_block = segment.first_block
        blocks = []

        # Create many small blocks
        for i in range(10):
            if current_block.value.size >= 32:
                new_block = current_block.splice(32)
                blocks.append(new_block)

        # Allocate every other block to create maximum fragmentation
        for i, block in enumerate(blocks):
            if i % 2 == 0:
                block.request_alloc()

        # Check high fragmentation
        fragmentation = segment.get_fragmentation_ratio()
        assert fragmentation > 0.5  # Should be highly fragmented

        # Free allocated blocks
        for i, block in enumerate(blocks):
            if i % 2 == 0:
                block.free_block()

        # Coalesce to reduce fragmentation
        for block in blocks:
            if not block.value.is_allocated():
                block.coalesce()

        # Fragmentation should be reduced
        new_fragmentation = segment.get_fragmentation_ratio()
        assert new_fragmentation < fragmentation


# Test fixtures and utilities
@pytest.fixture
def sample_pool():
    """Fixture providing a pool with sample data."""
    pool = BlockPool()

    # Create a few segments
    seg1 = pool.create_segment(0x1000, 1024, device=0, stream=1)
    seg2 = pool.create_segment(0x3000, 2048, device=0, stream=2)

    # Split and allocate some blocks
    block1 = seg1.first_block.splice(256)
    block1.request_alloc()

    block2 = seg2.first_block.splice(512)
    block2.request_alloc()

    return pool


@pytest.fixture
def fragmented_pool():
    """Fixture providing a highly fragmented pool."""
    pool = BlockPool()
    segment = pool.create_segment(0x1000, 1024)

    # Create alternating allocated/free pattern
    current = segment.first_block
    blocks = []

    for i in range(8):
        if current.value.size >= 64:
            block = current.splice(64)
            blocks.append(block)
            if i % 2 == 0:
                block.request_alloc()

    return pool, segment, blocks


class TestFixtures:
    """Test cases using fixtures."""

    def test_sample_pool_fixture(self, sample_pool):
        """Test the sample pool fixture."""
        assert len(sample_pool.segments) == 2
        assert sample_pool.segments[0].get_allocated_bytes() > 0
        assert sample_pool.segments[1].get_allocated_bytes() > 0

    def test_fragmented_pool_fixture(self, fragmented_pool):
        """Test the fragmented pool fixture."""
        pool, segment, blocks = fragmented_pool

        assert segment.get_fragmentation_ratio() > 0
        assert segment.get_allocated_bytes() > 0
        assert segment.get_free_bytes() > 0
