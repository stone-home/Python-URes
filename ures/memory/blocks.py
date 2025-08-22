from __future__ import annotations
import copy
import time
import traceback
import inspect
from typing import Any, Optional, Dict, List
from sortedcontainers import SortedSet
from dataclasses import dataclass, field
from ures.data_structure import NonCircularBiLink


@dataclass(slots=True)
class TraceInfo:
    """Represents trace information for memory operations"""

    timestamp_ns: int
    operation: (
        str  # "create", "alloc", "free_request", "free_complete", "split", "coalesce"
    )
    filename: Optional[str] = field(default=None)
    line_number: Optional[int] = field(default=None)
    function_name: Optional[str] = field(default=None)
    code_context: Optional[str] = field(default=None)
    stack_trace: Optional[List[str]] = field(default=None)
    additional_info: Optional[Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def capture_current_trace(
        cls,
        operation: str,
        stack_depth: int = 2,
        capture_full_stack: bool = False,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> "TraceInfo":
        """Capture current execution trace information"""
        timestamp_ns = time.time_ns()

        # Get current frame info
        frame_info = None
        try:
            current_frame = inspect.currentframe()
            # Go up the stack to find the calling frame
            for _ in range(stack_depth):
                if current_frame and current_frame.f_back:
                    current_frame = current_frame.f_back

            if current_frame:
                frame_info = inspect.getframeinfo(current_frame)
        except Exception:
            pass  # Fallback gracefully if frame inspection fails

        filename = frame_info.filename if frame_info else None
        line_number = frame_info.lineno if frame_info else None
        function_name = frame_info.function if frame_info else None
        code_context = (
            frame_info.code_context[0].strip()
            if frame_info and frame_info.code_context
            else None
        )

        # Capture full stack trace if requested
        stack_trace = None
        if capture_full_stack:
            try:
                stack_trace = traceback.format_stack()[
                    :-stack_depth
                ]  # Exclude current frames
            except Exception:
                stack_trace = None

        return cls(
            timestamp_ns=timestamp_ns,
            operation=operation,
            filename=filename,
            line_number=line_number,
            function_name=function_name,
            code_context=code_context,
            stack_trace=stack_trace,
            additional_info=additional_info or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Export trace data in dictionary form"""
        return {
            "timestamp_ns": self.timestamp_ns,
            "operation": self.operation,
            "filename": self.filename,
            "line_number": self.line_number,
            "function_name": self.function_name,
            "code_context": self.code_context,
            "stack_trace": self.stack_trace,
            "additional_info": self.additional_info,
        }


@dataclass(slots=True)
class MemoryInfo:
    """Represents a memory block within a segment"""

    addr: int
    size: int
    action: str = field(
        default="free"
    )  # "free", "alloc", "free_requested", "free_completed"
    allocated: bool = field(default=False)
    alloc_time_ns: Optional[int] = field(default=None)
    free_requested_time_ns: Optional[int] = field(default=None)
    free_completed_time_ns: Optional[int] = field(default=None)
    traces: List[TraceInfo] = field(default_factory=list)

    def is_allocated(self) -> bool:
        return self.allocated and self.action in ["alloc", "free_requested"]

    def is_free_requested(self) -> bool:
        return self.action == "free_requested"

    def is_free_completed(self) -> bool:
        return self.action == "free_completed" and not self.allocated

    def get_end_addr(self) -> int:
        """Get the end address of the block"""
        return self.addr + self.size

    def add_trace(self, trace: TraceInfo):
        """Add a trace to this memory info"""
        self.traces.append(trace)


@dataclass
class Segment:
    """Represents an original memory segment before any splitting"""

    segment_id: int
    start_addr: int
    original_size: int
    device: Optional[int] = field(default=None)
    stream: Optional[int] = field(default=None)
    creation_time_ns: Optional[int] = field(default=None)
    first_block: Optional[MemoryBlock] = field(default=None)
    traces: List[TraceInfo] = field(default_factory=list)

    def __post_init__(self):
        if self.creation_time_ns is None:
            self.creation_time_ns = time.time_ns()

    @property
    def end_addr(self) -> int:
        """Get the end address of the original segment"""
        return self.start_addr + self.original_size

    def add_trace(self, trace: TraceInfo):
        """Add a trace to this segment"""
        self.traces.append(trace)

    def get_blocks(self) -> List[MemoryBlock]:
        """Get all blocks belonging to this segment in address order"""
        if not self.first_block:
            return []

        blocks = []
        current = self.first_block.get_head()

        # Traverse the linked list and collect blocks from this segment
        while current:
            if hasattr(current, "segment_id") and current.segment_id == self.segment_id:
                blocks.append(current)
            current = current.next

        return sorted(blocks, key=lambda b: b.addr)

    def get_block_count(self) -> int:
        """Get the number of blocks this segment has been split into"""
        return len(self.get_blocks())

    def get_allocated_bytes(self) -> int:
        """Get total allocated bytes in this segment"""
        return sum(
            block.value.size
            for block in self.get_blocks()
            if block.value.is_allocated()
        )

    def get_free_bytes(self) -> int:
        """Get total free bytes in this segment"""
        return sum(
            block.value.size
            for block in self.get_blocks()
            if not block.value.is_allocated()
        )

    def get_utilization_ratio(self) -> float:
        """Get the utilization ratio (0.0 to 1.0)"""
        if self.original_size == 0:
            return 0.0
        return self.get_allocated_bytes() / self.original_size

    def get_fragmentation_ratio(self) -> float:
        """Get the fragmentation ratio (0.0 = no fragmentation, 1.0 = highly fragmented)"""
        block_count = self.get_block_count()
        if block_count <= 1:
            return 0.0
        return 1.0 - (1.0 / block_count)

    def is_fully_free(self) -> bool:
        """Check if all blocks in the segment are free"""
        return all(not block.value.is_allocated() for block in self.get_blocks())

    def is_fully_allocated(self) -> bool:
        """Check if all blocks in the segment are allocated"""
        return all(block.value.is_allocated() for block in self.get_blocks())

    def contains_address(self, addr: int, size: int = 1) -> bool:
        """Check if the specified address range is within the segment"""
        return self.start_addr <= addr and addr + size <= self.end_addr

    def to_dict(self) -> Dict[str, Any]:
        """Export segment data in dictionary form"""
        return {
            "segment_id": self.segment_id,
            "start_addr": self.start_addr,
            "start_addr_hex": hex(self.start_addr),
            "end_addr": self.end_addr,
            "end_addr_hex": hex(self.end_addr),
            "original_size": self.original_size,
            "device": self.device,
            "stream": self.stream,
            "creation_time_ns": self.creation_time_ns,
            "block_count": self.get_block_count(),
            "allocated_bytes": self.get_allocated_bytes(),
            "free_bytes": self.get_free_bytes(),
            "utilization_ratio": self.get_utilization_ratio(),
            "fragmentation_ratio": self.get_fragmentation_ratio(),
            "is_fully_free": self.is_fully_free(),
            "is_fully_allocated": self.is_fully_allocated(),
            "traces": [trace.to_dict() for trace in self.traces],
        }


class MemoryBlock(NonCircularBiLink):
    device: Optional[int] = field(default=None)  # Device ID for the memory block
    stream: Optional[int] = field(default=None)  # Stream only for CUDA
    pool: Optional[BlockPool] = field(default=None)

    def __init__(
        self,
        addr: int,
        size: int,
        device: Optional[int] = None,
        stream: Optional[int] = None,
        pool: Optional[BlockPool] = None,
        segment_id: Optional[int] = None,
        capture_trace: bool = True,
    ):
        """
        Create a memory block with the given address, size, stream, and time.

        Args:
                addr (int): The starting address of the memory block.
                size (int): The size of the memory block in bytes.
                device (int): The device identifier for the memory block.
                stream (int): The stream identifier for the memory block.
                pool (BlockPool): The pool this block belongs to.
                segment_id (int): The ID of the segment this block belongs to.
                capture_trace (bool): Whether to capture creation trace.

        Returns:
                None
        """
        # device or -1 causes the issue as 0 is treated as False
        self.device = device if device is not None else -1
        self.stream = stream
        self.pool = pool
        self.segment_id = segment_id  # NEW: Track which segment this block belongs to

        memory_info = MemoryInfo(copy.deepcopy(addr), copy.deepcopy(size))

        # Capture creation trace if requested
        if capture_trace:
            creation_trace = TraceInfo.capture_current_trace(
                operation="create",
                stack_depth=2,
                additional_info={
                    "addr": addr,
                    "size": size,
                    "device": device,
                    "stream": stream,
                    "segment_id": segment_id,
                },
            )
            memory_info.add_trace(creation_trace)

        super().__init__(memory_info)

    @property
    def value(self) -> MemoryInfo:
        return self._value

    @property
    def end_addr(self) -> int:
        """Get the end address of the block"""
        return self.value.get_end_addr()

    @property
    def is_head(self) -> bool:
        return self.prev is None

    @property
    def is_split(self) -> bool:
        return self.prev is not None or self.next is not None

    @property
    def addr(self) -> int:
        return self.value.addr

    @property
    def addr_hex(self) -> str:
        return hex(self.addr)

    @property
    def is_segment_start(self) -> bool:
        """Check if this block is the first block of a segment"""
        if self.pool and self.segment_id is not None:
            segment = self.pool.get_segment(self.segment_id)
            return segment and segment.first_block == self
        return False

    def request_alloc(self, time_ns: Optional[int] = None, capture_trace: bool = True):
        if self.value.action == "free":
            self.value.action = "alloc"
            self.value.allocated = True
            self.value.alloc_time_ns = time_ns or int(time.time_ns())

            # Capture allocation trace
            if capture_trace:
                alloc_trace = TraceInfo.capture_current_trace(
                    operation="alloc",
                    stack_depth=2,
                    additional_info={
                        "addr": self.addr,
                        "size": self.value.size,
                        "alloc_time_ns": self.value.alloc_time_ns,
                    },
                )
                self.value.add_trace(alloc_trace)

    def request_free(self, time_ns: Optional[int] = None, capture_trace: bool = True):
        """Request to free the memory block"""
        if self.value.action == "alloc":
            self.value.action = "free_requested"
            self.value.free_requested_time_ns = time_ns or int(time.time_ns())

            # Capture free request trace
            if capture_trace:
                free_request_trace = TraceInfo.capture_current_trace(
                    operation="free_request",
                    stack_depth=2,
                    additional_info={
                        "addr": self.addr,
                        "size": self.value.size,
                        "free_requested_time_ns": self.value.free_requested_time_ns,
                    },
                )
                self.value.add_trace(free_request_trace)

    def complete_free(self, time_ns: Optional[int] = None, capture_trace: bool = True):
        """Complete the free operation"""
        if self.value.action == "free_requested":
            self.value.action = "free_completed"
            self.value.allocated = False
            self.value.free_completed_time_ns = time_ns or int(time.time_ns())

            # Capture free completion trace
            if capture_trace:
                free_complete_trace = TraceInfo.capture_current_trace(
                    operation="free_complete",
                    stack_depth=2,
                    additional_info={
                        "addr": self.addr,
                        "size": self.value.size,
                        "free_completed_time_ns": self.value.free_completed_time_ns,
                    },
                )
                self.value.add_trace(free_complete_trace)

    def free_block(self, time_ns: Optional[int] = None, capture_trace: bool = True):
        """Immediately free the memory block (skips free_requested state)"""
        self.value.action = "free_completed"
        self.value.allocated = False
        self.value.free_completed_time_ns = time_ns or int(time.time_ns())

        # Capture immediate free trace
        if capture_trace:
            free_trace = TraceInfo.capture_current_trace(
                operation="free_immediate",
                stack_depth=2,
                additional_info={
                    "addr": self.addr,
                    "size": self.value.size,
                    "free_completed_time_ns": self.value.free_completed_time_ns,
                },
            )
            self.value.add_trace(free_trace)

    def force_reset_memory_info(self):
        mem_info = MemoryInfo(addr=self.addr, size=self.value.size)
        self._value = mem_info

    def insert_block(self, block: MemoryBlock) -> MemoryBlock:
        if self.value.is_allocated():
            raise MemoryError(
                f"Cannot insert a block into an allocated block {self.addr_hex} of size {self.value.size}"
            )
        if not self.contains_address(block.addr, block.value.size):
            raise ValueError(
                f"Cannot insert block {block.addr_hex} of size {block.value.size} into {self.addr_hex} of size {self.value.size}"
            )
        if block.is_split:
            # todo: handle split blocks
            raise ValueError("Cannot insert a split block into another block")
        if block.addr != self.addr:
            self.splice(block.addr - self.addr)
        return self.splice(block.value.size)

    def splice(
        self, memory_size: int, capture_trace: bool = True
    ) -> Optional[MemoryBlock]:
        """Split the block.

        Args:
                memory_size (int): The size of the memory.
                capture_trace (bool): Whether to capture split trace.

        Returns:
                MemoryBlock: The new block.
        """
        if self.value.is_allocated():
            raise MemoryError(f"Cannot split an allocated block()")
        if memory_size > self.value.size:
            raise ValueError(
                f"Cannot split a block of size {self.value.size} into a block of size {memory_size}"
            )
        if memory_size == self.value.size:
            if self.pool:
                if self in self.pool.blocks:
                    # Typically, the fetched block should be removed from pool in upper layer, which is allocator or
                    # Memory Manager. But just in case, we remove it here,ensuring consistency.
                    self.pool.blocks.remove(self)
            return self

        new_block = MemoryBlock(
            addr=self.value.addr,
            size=memory_size,
            device=self.device,
            stream=self.stream,
            pool=self.pool,
            segment_id=self.segment_id,  # NEW: Preserve segment ID
            capture_trace=capture_trace,
        )

        old_size = self.value.size
        self.value.size -= memory_size
        if self.addr is not None:
            self.value.addr += memory_size
        self.insert_before(new_block)

        # Capture split trace for both blocks
        if capture_trace:
            split_trace = TraceInfo.capture_current_trace(
                operation="split",
                stack_depth=2,
                additional_info={
                    "original_addr": new_block.addr,
                    "original_size": old_size,
                    "new_block_size": memory_size,
                    "remaining_block_addr": self.addr,
                    "remaining_block_size": self.value.size,
                },
            )
            self.value.add_trace(split_trace)
            new_block.value.add_trace(split_trace)

        if self.pool and self.segment_id is not None:
            segment = self.pool.get_segment(self.segment_id)
            if segment and segment.first_block == self:
                segment.first_block = new_block
        return new_block

    def coalesce(self, capture_trace: bool = True) -> MemoryBlock:
        """Coalesce the block."""
        if self.value.is_allocated():
            raise MemoryError(f"Cannot split an allocated block()")

        original_addr = self.addr
        original_size = self.value.size
        coalesced_blocks = []

        # --- Identify all blocks prepared to be removed ---
        blocks_to_merge = [self]

        _prev = None
        if (
            self.prev is not None
            and not self.prev.value.is_allocated()
            and hasattr(self.prev, "segment_id")
            and self.prev.segment_id == self.segment_id
        ):
            _prev = self.prev
            blocks_to_merge.append(_prev)

        _next = None
        if (
            self.next is not None
            and not self.next.value.is_allocated()
            and hasattr(self.next, "segment_id")
            and self.next.segment_id == self.segment_id
        ):
            _next = self.next
            blocks_to_merge.append(_next)

        # As only one block can be coalesced at a time, we just return if no coalesce is possible
        if len(blocks_to_merge) <= 1:
            if self.pool is not None:
                # ensure the block is valid in the pool
                if self in self.pool.blocks:
                    self.pool.blocks.remove(self)
                self.pool.blocks.add(self)
            return self

        # Remove all prepared blocks from pool first
        if self.pool is not None:
            for block in blocks_to_merge:
                if block in self.pool.blocks:
                    self.pool.blocks.remove(block)

        if _next:
            coalesced_blocks.append(("next", _next.addr, _next.value.size))
            self.value.size += _next.value.size
            _next.remove()

        if _prev:
            coalesced_blocks.append(("prev", _prev.addr, _prev.value.size))
            self.value.size += _prev.value.size
            self.value.addr = _prev.value.addr
            _prev.remove()

            if self.pool and self.segment_id is not None:
                segment = self.pool.get_segment(self.segment_id)
                if segment and segment.first_block == _prev:
                    segment.first_block = self

        # Capture coalesce trace
        if capture_trace and coalesced_blocks:
            coalesce_trace = TraceInfo.capture_current_trace(
                operation="coalesce",
                stack_depth=2,
                additional_info={
                    "original_addr": original_addr,
                    "original_size": original_size,
                    "final_addr": self.addr,
                    "final_size": self.value.size,
                    "coalesced_blocks": coalesced_blocks,
                },
            )
            self.value.add_trace(coalesce_trace)

        ## Typically, below two lines are not needed, but they ensure the block is in a consistent state.
        self.value.allocated = False
        self.value.action = "free"
        if self.pool is not None:
            # ensure the block is in the pool
            self.pool.insert_into_blocks(self)
        return self

    def contains_address(self, addr: int, size: int = 1) -> bool:
        """Check if the specified address range is within the segment"""
        return self.addr <= addr and addr + size <= self.end_addr

    def to_dict(self) -> Dict[str, Any]:
        """Export block data in dictionary form"""
        return {
            "addr": self.addr,
            "addr_hex": self.addr_hex,
            "end_addr": self.end_addr,
            "end_addr_hex": hex(self.end_addr),
            "size": self.value.size,
            "device": self.device,
            "stream": self.stream,
            "segment_id": self.segment_id,
            "is_allocated": self.value.is_allocated(),
            "is_segment_start": self.is_segment_start,
            "is_split": self.is_split,
            "action": self.value.action,
            "allocated": self.value.allocated,
            "alloc_time_ns": self.value.alloc_time_ns,
            "free_requested_time_ns": self.value.free_requested_time_ns,
            "free_completed_time_ns": self.value.free_completed_time_ns,
            "traces": [trace.to_dict() for trace in self.value.traces],
        }

    def __hash__(self):
        return hash((self.device, self.stream, self.value.size, self.addr))

    def __eq__(self, other: MemoryBlock):
        return (
            self.device == other.device
            and self.stream == other.stream
            and self.addr == other.addr
            and self.value.size == other.value.size
        )

    def __lt__(self, other: "MemoryBlock") -> bool:
        """Compare blocks for sorting (stream, size, then address)"""
        if not isinstance(other, MemoryBlock):
            return NotImplemented

        # Compare by stream first
        if self.stream != other.stream:
            # Handle None streams consistently
            if self.stream is None:
                return other.stream is not None
            if other.stream is None:
                return False
            return self.stream < other.stream

        # Then by size
        if self.value.size != other.value.size:
            return self.value.size < other.value.size

        # Finally by address
        return self.addr < other.addr

    def __le__(self, other):
        return self < other or self == other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other


class BlockPool:
    def __init__(self):
        self.blocks = SortedSet(
            key=lambda block: (block.stream, block.value.size, block.addr)
        )
        # NEW: Track segments for efficient listing
        self.segments: Dict[int, Segment] = {}
        self.next_segment_id: int = 0

    def _get_lower_bound_index(self, search_key: MemoryBlock) -> int:
        # Find the index of the first element that is >= search_key
        if search_key.addr is None:
            idx = self.blocks.bisect_key_left(
                (search_key.stream, search_key.value.size)
            )
        else:
            idx = self.blocks.bisect_key_left(
                (search_key.stream, search_key.value.size, search_key.addr)
            )
        return idx

    def lower_bound(
        self, search_key: MemoryBlock, releaseable=False
    ) -> Optional[MemoryBlock]:
        # Find the index of the first element that is >= search_key
        idx = self._get_lower_bound_index(search_key)
        if idx < len(self.blocks):
            if releaseable:
                _block: MemoryBlock = self.blocks[idx]
                while not (_block.prev is None and _block.next is None):
                    idx += 1
                    if idx >= len(self.blocks):
                        return None
                    _block = self.blocks[idx]
            else:
                return self.blocks[idx]
        else:
            return None  # No element >= search_key

    def is_end_block(self, search_key: MemoryBlock) -> bool:
        if search_key is None:
            return True
        idx = self._get_lower_bound_index(search_key)
        return idx >= len(self.blocks)

    def is_begin_block(self, search_key: MemoryBlock) -> bool:
        idx = self._get_lower_bound_index(search_key)
        return idx == 0

    def insert_into_blocks(self, block: MemoryBlock) -> None:
        self.blocks.add(block)

    def check_segment_overlap(
        self,
        start_addr: int,
        size: int,
        device: Optional[int] = None,
        stream: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Check if a proposed segment would overlap with any existing segments.

        Args:
                start_addr (int): Starting address of the proposed segment
                size (int): Size of the proposed segment
                device (Optional[int]): Device ID (optional filter)
                stream (Optional[int]): Stream ID (optional filter)

        Returns:
                Dict[str, Any]: Dictionary containing overlap information:
                        - 'has_overlap': bool indicating if any overlap exists
                        - 'overlapping_segments': list of overlapping segment details
                        - 'overlap_type': type of overlap ('none', 'partial', 'complete', 'contains', 'contained')
                        - 'safe_to_create': bool indicating if it's safe to create the segment
        """
        if size <= 0:
            return {
                "has_overlap": False,
                "overlapping_segments": [],
                "overlap_type": "none",
                "safe_to_create": False,
                "error": "Invalid size: must be greater than 0",
            }

        end_addr = start_addr + size
        overlapping_segments = []
        overlap_types = set()

        for segment in self.segments.values():
            # Filter by device and stream if specified
            if device is not None and segment.device != device:
                continue
            if stream is not None and segment.stream != stream:
                continue

            seg_start = segment.start_addr
            seg_end = segment.end_addr

            # Check for overlap
            if start_addr < seg_end and end_addr > seg_start:
                # Determine overlap type
                overlap_type = self._determine_overlap_type(
                    start_addr, end_addr, seg_start, seg_end
                )
                overlap_types.add(overlap_type)

                overlapping_segments.append(
                    {
                        "segment_id": segment.segment_id,
                        "start_addr": seg_start,
                        "start_addr_hex": hex(seg_start),
                        "end_addr": seg_end,
                        "end_addr_hex": hex(seg_end),
                        "size": segment.original_size,
                        "device": segment.device,
                        "stream": segment.stream,
                        "overlap_type": overlap_type,
                        "overlap_start": max(start_addr, seg_start),
                        "overlap_end": min(end_addr, seg_end),
                        "overlap_size": min(end_addr, seg_end)
                        - max(start_addr, seg_start),
                    }
                )

        has_overlap = len(overlapping_segments) > 0

        # Determine overall overlap type
        if not overlap_types:
            overall_overlap_type = "none"
        elif len(overlap_types) == 1:
            overall_overlap_type = list(overlap_types)[0]
        else:
            overall_overlap_type = "multiple"

        return {
            "has_overlap": has_overlap,
            "overlapping_segments": overlapping_segments,
            "overlap_type": overall_overlap_type,
            "safe_to_create": not has_overlap,
            "proposed_segment": {
                "start_addr": start_addr,
                "start_addr_hex": hex(start_addr),
                "end_addr": end_addr,
                "end_addr_hex": hex(end_addr),
                "size": size,
                "device": device,
                "stream": stream,
            },
        }

    def _determine_overlap_type(
        self, start1: int, end1: int, start2: int, end2: int
    ) -> str:
        """
        Determine the type of overlap between two address ranges.

        Args:
                start1, end1: First range (proposed segment)
                start2, end2: Second range (existing segment)

        Returns:
                str: Type of overlap ('partial', 'complete', 'contains', 'contained')
        """
        if start1 == start2 and end1 == end2:
            return "complete"  # Exact same range
        elif start1 <= start2 and end1 >= end2:
            return "contains"  # Proposed segment contains existing segment
        elif start1 >= start2 and end1 <= end2:
            return "contained"  # Proposed segment is contained in existing segment
        else:
            return "partial"  # Partial overlap

    def find_safe_address_range(
        self,
        size: int,
        device: Optional[int] = None,
        stream: Optional[int] = None,
        min_addr: int = 0x1000,
        max_addr: int = 0xFFFFFFFF,
        alignment: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """
        Find a safe address range where a new segment can be created without overlap.

        Args:
                size (int): Required size for the new segment
                device (Optional[int]): Device ID filter
                stream (Optional[int]): Stream ID filter
                min_addr (int): Minimum address to consider
                max_addr (int): Maximum address to consider
                alignment (int): Address alignment requirement

        Returns:
                Optional[Dict[str, Any]]: Safe address range info or None if no space found
        """
        if size <= 0:
            return None

        # Get all relevant segments sorted by address
        relevant_segments = []
        for segment in self.segments.values():
            if device is not None and segment.device != device:
                continue
            if stream is not None and segment.stream != stream:
                continue
            relevant_segments.append((segment.start_addr, segment.end_addr))

        relevant_segments.sort()

        # Align the minimum address
        current_addr = ((min_addr + alignment - 1) // alignment) * alignment

        # Check if we can fit before the first segment
        if relevant_segments and current_addr + size <= relevant_segments[0][0]:
            return {
                "start_addr": current_addr,
                "start_addr_hex": hex(current_addr),
                "end_addr": current_addr + size,
                "end_addr_hex": hex(current_addr + size),
                "size": size,
                "location": "before_first_segment",
            }

        # Check gaps between segments
        for i in range(len(relevant_segments) - 1):
            gap_start = relevant_segments[i][1]  # End of current segment
            gap_end = relevant_segments[i + 1][0]  # Start of next segment

            # Align the gap start
            aligned_start = ((gap_start + alignment - 1) // alignment) * alignment

            if aligned_start + size <= gap_end:
                return {
                    "start_addr": aligned_start,
                    "start_addr_hex": hex(aligned_start),
                    "end_addr": aligned_start + size,
                    "end_addr_hex": hex(aligned_start + size),
                    "size": size,
                    "location": f"gap_after_segment_{i}",
                    "gap_start": gap_start,
                    "gap_end": gap_end,
                }

        # Check if we can fit after the last segment
        if relevant_segments:
            last_end = relevant_segments[-1][1]
            aligned_start = ((last_end + alignment - 1) // alignment) * alignment

            if aligned_start + size <= max_addr:
                return {
                    "start_addr": aligned_start,
                    "start_addr_hex": hex(aligned_start),
                    "end_addr": aligned_start + size,
                    "end_addr_hex": hex(aligned_start + size),
                    "size": size,
                    "location": "after_last_segment",
                }
        else:
            # No existing segments, use min_addr
            if current_addr + size <= max_addr:
                return {
                    "start_addr": current_addr,
                    "start_addr_hex": hex(current_addr),
                    "end_addr": current_addr + size,
                    "end_addr_hex": hex(current_addr + size),
                    "size": size,
                    "location": "first_segment",
                }

        return None  # No safe space found

    def create_segment(
        self,
        start_addr: int,
        size: int,
        device: Optional[int] = None,
        stream: Optional[int] = None,
        capture_trace: bool = True,
    ) -> Segment:
        """Create a new segment and its initial block"""
        overlap_info = self.check_segment_overlap(start_addr, size, device, stream)
        if overlap_info["has_overlap"]:
            raise MemoryError(
                f"Cannot create segment at {hex(start_addr)} of size {size}: Overlaps with existing segments: {overlap_info['overlapping_segments']}"
            )

        segment_id = self.next_segment_id
        self.next_segment_id += 1

        # Create the segment
        segment = Segment(
            segment_id=segment_id,
            start_addr=start_addr,
            original_size=size,
            device=device,
            stream=stream,
        )

        # Capture segment creation trace
        if capture_trace:
            creation_trace = TraceInfo.capture_current_trace(
                operation="create_segment",
                stack_depth=2,
                additional_info={
                    "segment_id": segment_id,
                    "start_addr": start_addr,
                    "size": size,
                    "device": device,
                    "stream": stream,
                },
            )
            segment.add_trace(creation_trace)

        # Create the initial block
        block = MemoryBlock(
            addr=start_addr,
            size=size,
            device=device,
            stream=stream,
            pool=self,
            segment_id=segment_id,
            capture_trace=capture_trace,
        )

        # Link them
        segment.first_block = block
        self.segments[segment_id] = segment
        self.insert_into_blocks(block)

        return segment

    def get_segment(self, segment_id: int) -> Optional[Segment]:
        """Get segment by ID"""
        return self.segments.get(segment_id)

    def list_all_segments(self) -> List[Dict[str, Any]]:
        """Swift listing of all segments with original information - O(segments) complexity"""
        segment_list = []
        for segment in self.segments.values():
            segment_list.append(segment.to_dict())

        return sorted(segment_list, key=lambda x: x["start_addr"])

    def list_blocks_in_segment(self, segment_id: int) -> List[Dict[str, Any]]:
        """List all blocks within a specific segment in address order"""
        segment = self.get_segment(segment_id)
        if not segment:
            return []

        blocks_info = []
        for block in segment.get_blocks():
            blocks_info.append(block.to_dict())

        return blocks_info

    def get_segment_by_address(self, addr: int) -> Optional[Segment]:
        """Find which segment contains the given address"""
        for segment in self.segments.values():
            if segment.contains_address(addr):
                return segment
        return None

    def remove_segment(self, segment_id: int, force: bool = True) -> bool:
        """Remove a segment and all its blocks"""
        segment = self.get_segment(segment_id)
        if not segment:
            return False

        # Remove all blocks belonging to this segment
        blocks_to_remove = segment.get_blocks()
        if not force:
            if len(blocks_to_remove) > 0:
                for block in blocks_to_remove:
                    if block.value.is_allocated():
                        ## Cannot remove segment if any block is allocated
                        return False

        for block in blocks_to_remove:
            if block in self.blocks:
                self.blocks.remove(block)
            block.remove()  # Remove from linked list

        # Remove the segment
        del self.segments[segment_id]
        return True

    def get_traces_by_operation(self, operation: str) -> List[Dict[str, Any]]:
        """Get all traces of a specific operation across all segments and blocks"""
        all_traces = []

        # Get traces from all segments
        for segment in self.segments.values():
            for trace in segment.traces:
                if trace.operation == operation:
                    trace_data = trace.to_dict()
                    trace_data["source_type"] = "segment"
                    trace_data["source_id"] = segment.segment_id
                    all_traces.append(trace_data)

            # Get traces from all blocks in this segment
            for block in segment.get_blocks():
                for trace in block.value.traces:
                    if trace.operation == operation:
                        trace_data = trace.to_dict()
                        trace_data["source_type"] = "block"
                        trace_data["source_addr"] = block.addr
                        trace_data["source_segment_id"] = segment.segment_id
                        all_traces.append(trace_data)

        return sorted(all_traces, key=lambda x: x["timestamp_ns"])

    def get_all_traces(self) -> List[Dict[str, Any]]:
        """Get all traces from all segments and blocks, sorted by timestamp"""
        all_traces = []

        # Get traces from all segments
        for segment in self.segments.values():
            for trace in segment.traces:
                trace_data = trace.to_dict()
                trace_data["source_type"] = "segment"
                trace_data["source_id"] = segment.segment_id
                all_traces.append(trace_data)

            # Get traces from all blocks in this segment
            for block in segment.get_blocks():
                for trace in block.value.traces:
                    trace_data = trace.to_dict()
                    trace_data["source_type"] = "block"
                    trace_data["source_addr"] = block.addr
                    trace_data["source_segment_id"] = segment.segment_id
                    all_traces.append(trace_data)

        return sorted(all_traces, key=lambda x: x["timestamp_ns"])

    def analyze_memory_patterns(self) -> Dict[str, Any]:
        """Analyze memory allocation patterns based on traces"""
        all_traces = self.get_all_traces()

        operations_count = {}
        operation_timings = {}

        for trace in all_traces:
            op = trace["operation"]
            operations_count[op] = operations_count.get(op, 0) + 1

            if op not in operation_timings:
                operation_timings[op] = []
            operation_timings[op].append(trace["timestamp_ns"])

        # Calculate operation frequency and timing patterns
        analysis = {
            "total_operations": len(all_traces),
            "operations_count": operations_count,
            "most_frequent_operation": (
                max(operations_count.items(), key=lambda x: x[1])
                if operations_count
                else None
            ),
            "operation_timeline": {
                op: {"first": min(times), "last": max(times), "count": len(times)}
                for op, times in operation_timings.items()
            },
        }

        return analysis

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get overall memory summary across all segments"""
        total_segments = len(self.segments)
        total_original_size = sum(seg.original_size for seg in self.segments.values())
        total_allocated = sum(
            seg.get_allocated_bytes() for seg in self.segments.values()
        )
        total_free = total_original_size - total_allocated
        total_blocks = sum(seg.get_block_count() for seg in self.segments.values())

        return {
            "total_segments": total_segments,
            "total_original_size": total_original_size,
            "total_allocated_bytes": total_allocated,
            "total_free_bytes": total_free,
            "total_blocks": total_blocks,
            "overall_utilization": (
                total_allocated / total_original_size
                if total_original_size > 0
                else 0.0
            ),
            "average_fragmentation": (
                sum(seg.get_fragmentation_ratio() for seg in self.segments.values())
                / total_segments
                if total_segments > 0
                else 0.0
            ),
        }

    def print_memory_status(self):
        """Print detailed memory status for debugging"""
        print("=== BlockPool Memory Status ===")
        summary = self.get_memory_summary()
        print(f"Total segments: {summary['total_segments']}")
        print(f"Total original size: {summary['total_original_size']} bytes")
        print(f"Total allocated: {summary['total_allocated_bytes']} bytes")
        print(f"Total free: {summary['total_free_bytes']} bytes")
        print(f"Total blocks: {summary['total_blocks']}")
        print(f"Overall utilization: {summary['overall_utilization']:.1%}")
        print(f"Average fragmentation: {summary['average_fragmentation']:.1%}")

        print("\n=== Segments ===")
        for segment_info in self.list_all_segments():
            print(f"\nSegment {segment_info['segment_id']}:")
            print(f"  Address: {segment_info['start_addr_hex']}")
            print(f"  Device/Stream: {segment_info['device']}/{segment_info['stream']}")
            print(f"  Original size: {segment_info['original_size']} bytes")
            print(f"  Allocated: {segment_info['allocated_bytes']} bytes")
            print(f"  Free: {segment_info['free_bytes']} bytes")
            print(f"  Blocks: {segment_info['block_count']}")
            print(f"  Utilization: {segment_info['utilization_ratio']:.1%}")
            print(f"  Fragmentation: {segment_info['fragmentation_ratio']:.1%}")

            # Show blocks
            blocks = self.list_blocks_in_segment(segment_info["segment_id"])
            for block in blocks:
                status = "ALLOC" if block["is_allocated"] else "FREE"
                start_marker = " [SEGMENT_START]" if block["is_segment_start"] else ""
                print(
                    f"    {block['addr_hex']}: {block['size']} bytes ({status}){start_marker}"
                )


# Example usage demonstrating the refactored code
if __name__ == "__main__":
    # Create a pool and add some segments
    pool = BlockPool()

    # Create segments (this would typically be done when you get memory from the system)
    seg1 = pool.create_segment(0x1000, 1024, device=0, stream=1)  # 1KB segment
    seg2 = pool.create_segment(0x2000, 2048, device=0, stream=2)  # 2KB segment

    print("Initial state:")
    pool.print_memory_status()

    # Simulate some allocations by splitting blocks
    block1 = seg1.first_block.splice(256)  # Split first segment
    block1.request_alloc()

    block2 = seg2.first_block.splice(512)  # Split second segment
    block2.request_alloc()

    # Split again
    remaining_block = seg1.get_blocks()[1]  # Get the remaining block
    block3 = remaining_block.splice(300)
    block3.request_alloc()

    print(f"\nAfter allocations and splits:")
    pool.print_memory_status()

    # Demonstrate swift segment listing
    print("\n=== Traces Analysis ===")
    analysis = pool.analyze_memory_patterns()
    print(f"Total operations: {analysis['total_operations']}")
    print(f"Operations count: {analysis['operations_count']}")
    if analysis["most_frequent_operation"]:
        print(
            f"Most frequent operation: {analysis['most_frequent_operation'][0]} ({analysis['most_frequent_operation'][1]} times)"
        )

    # Show some example traces
    print("\n=== Recent Traces ===")
    recent_traces = pool.get_all_traces()[-5:]  # Last 5 traces
    for trace in recent_traces:
        print(
            f"  {trace['operation']} at {trace['function_name']}:{trace['line_number']} - {trace['code_context']}"
        )
        if trace.get("additional_info"):
            print(f"    Additional info: {trace['additional_info']}")
