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
    """
    Represents trace information for memory operations.

    This class captures execution context and timing information for debugging
    and profiling memory allocation/deallocation operations. It records stack
    traces, function names, and additional metadata to help track memory usage patterns.

    Attributes:
            timestamp_ns: Timestamp in nanoseconds when the operation occurred
            operation: Type of operation ("create", "alloc", "free_request", etc.)
            filename: Source file where the operation was initiated
            line_number: Line number in the source file
            function_name: Name of the function that initiated the operation
            code_context: The actual line of code that was executed
            stack_trace: Full stack trace if capture_full_stack was enabled
            additional_info: Dictionary containing operation-specific metadata
    """

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
        """
        Capture current execution trace information.

        This method inspects the current call stack to gather context information
        about where a memory operation was initiated. It's useful for debugging
        memory leaks and understanding allocation patterns.

        Args:
                operation: String describing the type of operation being traced
                stack_depth: How many frames up the stack to look for the caller
                capture_full_stack: Whether to capture the complete stack trace
                additional_info: Optional dictionary with operation-specific data

        Returns:
                TraceInfo object containing the captured trace information
        """
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
        """
        Export trace data in dictionary form.

        Converts the TraceInfo object to a dictionary format suitable for
        JSON serialization, logging, or analysis tools.

        Returns:
                Dictionary containing all trace information
        """
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
    """
    Represents a memory block within a segment.

    This class tracks the state and lifecycle of a memory block, including
    allocation status, timing information, and operation traces. It maintains
    the core information needed for memory management operations.

    Attributes:
            addr: Starting address of the memory block
            size: Size of the memory block in bytes
            action: Current state ("free", "alloc", "free_requested", "free_completed")
            allocated: Boolean flag indicating if the block is currently allocated
            alloc_time_ns: Timestamp when the block was allocated
            free_requested_time_ns: Timestamp when free was requested
            free_completed_time_ns: Timestamp when free was completed
            traces: List of TraceInfo objects tracking operations on this block
    """

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
        """
        Check if the memory block is currently allocated.

        A block is considered allocated if it's marked as allocated and
        its action is either "alloc" or "free_requested" (pending free).

        Returns:
                True if the block is allocated, False otherwise
        """
        return self.allocated and self.action in ["alloc", "free_requested"]

    def is_free_requested(self) -> bool:
        """
        Check if a free operation has been requested but not completed.

        Returns:
                True if free has been requested but not completed
        """
        return self.action == "free_requested"

    def is_free_completed(self) -> bool:
        """
        Check if the memory block has been completely freed.

        Returns:
                True if the block is fully freed and available for reuse
        """
        return self.action == "free_completed" and not self.allocated

    def get_end_addr(self) -> int:
        """
        Get the end address of the block.

        Calculates the first address after the end of this memory block.

        Returns:
                The address immediately following this block
        """
        return self.addr + self.size

    def add_trace(self, trace: TraceInfo):
        """
        Add a trace to this memory info.

        Appends a TraceInfo object to track operations performed on this block.

        Args:
                trace: TraceInfo object containing operation details
        """
        self.traces.append(trace)


@dataclass
class Segment:
    """
    Represents an original memory segment before any splitting.

    A segment represents a contiguous block of memory that was originally
    allocated from the system. It can be split into multiple smaller blocks
    but maintains information about the original allocation for tracking
    fragmentation and utilization.

    Attributes:
            segment_id: Unique identifier for this segment
            start_addr: Starting address of the original segment
            original_size: Original size of the segment when first created
            device: Device ID where this memory resides (e.g., GPU device)
            stream: Stream ID for CUDA operations
            creation_time_ns: Timestamp when the segment was created
            first_block: Reference to the first block in the linked list
            traces: List of TraceInfo objects for segment-level operations
    """

    segment_id: int
    start_addr: int
    original_size: int
    device: Optional[int] = field(default=None)
    stream: Optional[int] = field(default=None)
    creation_time_ns: Optional[int] = field(default=None)
    first_block: Optional[MemoryBlock] = field(default=None)
    traces: List[TraceInfo] = field(default_factory=list)

    def __post_init__(self):
        """Set creation timestamp if not provided."""
        if self.creation_time_ns is None:
            self.creation_time_ns = time.time_ns()

    @property
    def end_addr(self) -> int:
        """
        Get the end address of the original segment.

        Returns:
                The address immediately following the original segment
        """
        return self.start_addr + self.original_size

    def add_trace(self, trace: TraceInfo):
        """
        Add a trace to this segment.

        Args:
                trace: TraceInfo object containing operation details
        """
        self.traces.append(trace)

    def get_blocks(self) -> List[MemoryBlock]:
        """
        Get all blocks belonging to this segment in address order.

        Traverses the linked list starting from first_block and collects
        all blocks that belong to this segment, then sorts them by address.

        Returns:
                List of MemoryBlock objects sorted by address
        """
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
        """
        Get the number of blocks this segment has been split into.

        Returns:
                Number of blocks in this segment
        """
        return len(self.get_blocks())

    def get_allocated_bytes(self) -> int:
        """
        Get total allocated bytes in this segment.

        Sums up the sizes of all allocated blocks within this segment.

        Returns:
                Total bytes currently allocated in this segment
        """
        return sum(
            block.value.size
            for block in self.get_blocks()
            if block.value.is_allocated()
        )

    def get_free_bytes(self) -> int:
        """
        Get total free bytes in this segment.

        Sums up the sizes of all free blocks within this segment.

        Returns:
                Total bytes currently free in this segment
        """
        return sum(
            block.value.size
            for block in self.get_blocks()
            if not block.value.is_allocated()
        )

    def get_utilization_ratio(self) -> float:
        """
        Get the utilization ratio (0.0 to 1.0).

        Calculates what fraction of the segment is currently allocated.

        Returns:
                Ratio of allocated bytes to original segment size
        """
        if self.original_size == 0:
            return 0.0
        return self.get_allocated_bytes() / self.original_size

    def get_fragmentation_ratio(self) -> float:
        """
        Get the fragmentation ratio (0.0 = no fragmentation, 1.0 = highly fragmented).

        Measures how fragmented the segment is based on the number of blocks.
        More blocks indicate higher fragmentation.

        Returns:
                Fragmentation ratio where 0.0 means no fragmentation
        """
        block_count = self.get_block_count()
        if block_count <= 1:
            return 0.0
        return 1.0 - (1.0 / block_count)

    def is_fully_free(self) -> bool:
        """
        Check if all blocks in the segment are free.

        Returns:
                True if every block in the segment is free
        """
        return all(not block.value.is_allocated() for block in self.get_blocks())

    def is_fully_allocated(self) -> bool:
        """
        Check if all blocks in the segment are allocated.

        Returns:
                True if every block in the segment is allocated
        """
        return all(block.value.is_allocated() for block in self.get_blocks())

    def contains_address(self, addr: int, size: int = 1) -> bool:
        """
        Check if the specified address range is within the segment.

        Args:
                addr: Starting address to check
                size: Size of the range to check

        Returns:
                True if the entire range is within the segment bounds
        """
        return self.start_addr <= addr and addr + size <= self.end_addr

    def to_dict(self) -> Dict[str, Any]:
        """
        Export segment data in dictionary form.

        Creates a comprehensive dictionary representation of the segment
        including all metrics and trace information.

        Returns:
                Dictionary containing complete segment information
        """
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
    """
    Represents a memory block that can be split, allocated, and coalesced.

    This class extends NonCircularBiLink to implement a doubly-linked list
    of memory blocks. Each block represents a contiguous region of memory
    that can be allocated to users or kept free for future allocations.
    Blocks can be split into smaller pieces or coalesced with adjacent
    free blocks to reduce fragmentation.

    Attributes:
            device: Device ID for the memory block (e.g., GPU device number)
            stream: Stream ID for CUDA operations
            pool: Reference to the BlockPool that manages this block
            segment_id: ID of the segment this block belongs to
    """

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
        """Get the MemoryInfo object associated with this block."""
        return self._value

    @property
    def end_addr(self) -> int:
        """Get the end address of the block."""
        return self.value.get_end_addr()

    @property
    def is_head(self) -> bool:
        """Check if this block is the head of the linked list."""
        return self.prev is None

    @property
    def is_split(self) -> bool:
        """Check if this block is part of a split (has neighbors in linked list)."""
        return self.prev is not None or self.next is not None

    @property
    def addr(self) -> int:
        """Get the starting address of the block."""
        return self.value.addr

    @property
    def addr_hex(self) -> str:
        """Get the starting address of the block in hexadecimal format."""
        return hex(self.addr)

    @property
    def is_segment_start(self) -> bool:
        """
        Check if this block is the first block of a segment.

        Returns:
                True if this block is the first block of its segment
        """
        if self.pool and self.segment_id is not None:
            segment = self.pool.get_segment(self.segment_id)
            return segment and segment.first_block == self
        return False

    def request_alloc(self, time_ns: Optional[int] = None, capture_trace: bool = True):
        """
        Request allocation of this memory block.

        Marks the block as allocated if it's currently free. Updates the
        allocation timestamp and captures trace information.

        Args:
                time_ns: Optional timestamp for the allocation (uses current time if None)
                capture_trace: Whether to capture trace information for this operation
        """
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
        """
        Request to free the memory block.

        Initiates the free process by marking the block as "free_requested".
        This allows for asynchronous freeing patterns where the actual free
        operation might be deferred.

        Args:
                time_ns: Optional timestamp for the free request
                capture_trace: Whether to capture trace information for this operation
        """
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
        """
        Complete the free operation.

        Finalizes the freeing process for blocks that were previously marked
        as "free_requested". This completes the asynchronous free pattern.

        Args:
                time_ns: Optional timestamp for the free completion
                capture_trace: Whether to capture trace information for this operation
        """
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
        """
        Immediately free the memory block (skips free_requested state).

        Performs an immediate synchronous free operation, bypassing the
        "free_requested" intermediate state.

        Args:
                time_ns: Optional timestamp for the free operation
                capture_trace: Whether to capture trace information for this operation
        """
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
        """
        Force reset the memory info to a clean state.

        Creates a new MemoryInfo object with the same address and size,
        effectively clearing all allocation history and traces.
        """
        mem_info = MemoryInfo(addr=self.addr, size=self.value.size)
        self._value = mem_info

    def insert_block(self, block: MemoryBlock) -> MemoryBlock:
        """
        Insert a block into this block's address space.

        This method handles the complex operation of inserting a new block
        within the address range of this block. It performs necessary
        splitting operations to accommodate the new block.

        Args:
                block: The MemoryBlock to insert

        Returns:
                The MemoryBlock that was created or modified during insertion

        Raises:
                MemoryError: If trying to insert into an allocated block
                ValueError: If the block doesn't fit or is already split
        """
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
            freed_block = self.splice(block.addr - self.addr)
            if self.pool:
                self.pool.blocks.add(freed_block)

        if block.end_addr == self.end_addr:
            return self
        return self.splice(block.value.size)

    # def insert_block(self, block: MemoryBlock) -> MemoryBlock:
    # 	"""
    # 	Insert a block into this block's address space.
    #
    # 	This method handles the complex operation of inserting a new block
    # 	within the address range of this block. It performs necessary
    # 	splitting operations to accommodate the new block.
    #
    # 	Args:
    # 		block: The MemoryBlock to insert
    #
    # 	Returns:
    # 		The MemoryBlock that was created or modified during insertion
    #
    # 	Raises:
    # 		MemoryError: If trying to insert into an allocated block
    # 		ValueError: If the block doesn't fit or is already split
    # 	"""
    # 	if self.value.is_allocated():
    # 		raise MemoryError(
    # 			f"Cannot insert a block into an allocated block {self.addr_hex} of size {self.value.size}")
    # 	if not self.contains_address(block.addr, block.value.size):
    # 		raise ValueError(
    # 			f"Cannot insert block {block.addr_hex} of size {block.value.size} into {self.addr_hex} of size {self.value.size}"
    # 		)
    # 	if block.is_split:
    # 		# todo: handle split blocks
    # 		raise ValueError("Cannot insert a split block into another block")
    # 	if block.addr != self.addr:
    # 		self.splice(block.addr - self.addr)
    # 	return self.splice(block.value.size)

    def splice(
        self, memory_size: int, capture_trace: bool = True
    ) -> Optional[MemoryBlock]:
        """
        Split the block.

        Creates a new block of the specified size from the beginning of this block
        and adjusts this block to represent the remaining memory. This is the
        fundamental operation for memory allocation.

        Args:
                memory_size (int): The size of the memory to split off.
                capture_trace (bool): Whether to capture split trace.

        Returns:
                MemoryBlock: The new block that was created.

        Raises:
                MemoryError: If trying to split an allocated block
                ValueError: If the requested size is larger than available
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
        """
        Coalesce the block with adjacent free blocks.

        Merges this block with any adjacent free blocks in the same segment
        to reduce fragmentation. This is a critical operation for maintaining
        memory efficiency.

        Args:
                capture_trace: Whether to capture trace information for this operation

        Returns:
                The coalesced MemoryBlock (may be self or a neighboring block)

        Raises:
                MemoryError: If trying to coalesce an allocated block
        """
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
        """
        Check if the specified address range is within the segment.

        Args:
                addr: Starting address to check
                size: Size of the range to check (default: 1)

        Returns:
                True if the entire range is within this block's bounds
        """
        return self.addr <= addr and addr + size <= self.end_addr

    def to_dict(self) -> Dict[str, Any]:
        """
        Export block data in dictionary form.

        Creates a comprehensive dictionary representation of the block
        including all state information and traces.

        Returns:
                Dictionary containing complete block information
        """
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
        """
        Generate hash for the MemoryBlock.

        Creates a hash based on device, stream, size, and address for use
        in sets and dictionaries.

        Returns:
                Hash value for this block
        """
        return hash((self.device, self.stream, self.value.size, self.addr))

    def __eq__(self, other: MemoryBlock):
        """
        Check equality with another MemoryBlock.

        Two blocks are considered equal if they have the same device, stream,
        address, and size.

        Args:
                other: Another MemoryBlock to compare with

        Returns:
                True if blocks are equal, False otherwise
        """
        return (
            self.device == other.device
            and self.stream == other.stream
            and self.addr == other.addr
            and self.value.size == other.value.size
        )

    def __lt__(self, other: "MemoryBlock") -> bool:
        """
        Compare blocks for sorting (stream, size, then address).

        Implements ordering for MemoryBlocks used by SortedSet. Blocks are
        sorted first by stream, then by size, then by address.

        Args:
                other: Another MemoryBlock to compare with

        Returns:
                True if this block should come before the other in sorted order
        """
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
        """Less than or equal comparison."""
        return self < other or self == other

    def __gt__(self, other):
        """Greater than comparison."""
        return not self <= other

    def __ge__(self, other):
        """Greater than or equal comparison."""
        return not self < other


class BlockPool:
    """
    Manages a pool of memory blocks and segments.

    BlockPool is the main memory management class that maintains collections
    of memory blocks and segments. It provides functionality for creating
    segments, tracking memory usage, analyzing allocation patterns, and
    detecting memory overlaps. The pool uses a SortedSet for efficient
    block lookups and maintains segment metadata for comprehensive memory
    analysis.

    Attributes:
            blocks: SortedSet of MemoryBlock objects sorted by (stream, size, address)
            segments: Dictionary mapping segment IDs to Segment objects
            next_segment_id: Counter for generating unique segment IDs
    """

    def __init__(self):
        """
        Initialize a new BlockPool.

        Creates empty collections for blocks and segments, and initializes
        the segment ID counter.
        """
        self.blocks = SortedSet(
            key=lambda block: (block.stream, block.value.size, block.addr)
        )
        # NEW: Track segments for efficient listing
        self.segments: Dict[int, Segment] = {}
        self.next_segment_id: int = 0

    def _get_lower_bound_index(self, search_key: MemoryBlock) -> int:
        """
        Find the index of the first element that is >= search_key.

        Uses binary search to efficiently locate the insertion point or
        first matching element in the sorted blocks collection.

        Args:
                search_key: MemoryBlock to search for

        Returns:
                Index of the first element >= search_key
        """
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
        """
        Find the first block that is >= search_key.

        Performs efficient lookup in the sorted blocks collection. Can optionally
        filter for blocks that are not part of a split (releaseable blocks).

        Args:
                search_key: MemoryBlock to search for
                releaseable: If True, only return blocks that are not split

        Returns:
                First MemoryBlock >= search_key, or None if not found
        """
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
        """
        Check if search_key would be at the end of the sorted collection.

        Args:
                search_key: MemoryBlock to check

        Returns:
                True if search_key would be inserted at the end
        """
        if search_key is None:
            return True
        idx = self._get_lower_bound_index(search_key)
        return idx >= len(self.blocks)

    def is_begin_block(self, search_key: MemoryBlock) -> bool:
        """
        Check if search_key would be at the beginning of the sorted collection.

        Args:
                search_key: MemoryBlock to check

        Returns:
                True if search_key would be inserted at the beginning
        """
        idx = self._get_lower_bound_index(search_key)
        return idx == 0

    def insert_into_blocks(self, block: MemoryBlock) -> None:
        """
        Insert a block into the sorted blocks collection.

        Args:
                block: MemoryBlock to insert
        """
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

        Analyzes the proposed segment against all existing segments to detect
        any address space conflicts. This is crucial for preventing memory
        corruption and ensuring safe segment creation.

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

        Searches through the address space to find a suitable location for
        a new segment of the specified size. Considers alignment requirements
        and avoids existing segments.

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
        """
        Create a new segment and its initial block.

        Creates a new memory segment with the specified parameters after
        checking for overlaps. This is the primary method for adding new
        memory regions to the pool.

        Args:
                start_addr: Starting address for the new segment
                size: Size of the segment in bytes
                device: Optional device ID
                stream: Optional stream ID
                capture_trace: Whether to capture creation traces

        Returns:
                The newly created Segment object

        Raises:
                MemoryError: If the segment would overlap with existing segments
        """
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
        """
        Get segment by ID.

        Args:
                segment_id: ID of the segment to retrieve

        Returns:
                Segment object if found, None otherwise
        """
        return self.segments.get(segment_id)

    def list_all_segments(self) -> List[Dict[str, Any]]:
        """
        Swift listing of all segments with original information - O(segments) complexity.

        Provides an efficient way to get summary information about all segments
        without traversing the block linked lists.

        Returns:
                List of dictionaries containing segment information, sorted by address
        """
        segment_list = []
        for segment in self.segments.values():
            segment_list.append(segment.to_dict())

        return sorted(segment_list, key=lambda x: x["start_addr"])

    def list_blocks_in_segment(self, segment_id: int) -> List[Dict[str, Any]]:
        """
        List all blocks within a specific segment in address order.

        Args:
                segment_id: ID of the segment to examine

        Returns:
                List of dictionaries containing block information
        """
        segment = self.get_segment(segment_id)
        if not segment:
            return []

        blocks_info = []
        for block in segment.get_blocks():
            blocks_info.append(block.to_dict())

        return blocks_info

    def get_segment_by_address(self, addr: int) -> Optional[Segment]:
        """
        Find which segment contains the given address.

        Args:
                addr: Address to search for

        Returns:
                Segment containing the address, or None if not found
        """
        for segment in self.segments.values():
            if segment.contains_address(addr):
                return segment
        return None

    def remove_segment(self, segment_id: int, force: bool = True) -> bool:
        """
        Remove a segment and all its blocks.

        Removes a segment and all associated blocks from the pool. Can optionally
        check for allocated blocks before removal.

        Args:
                segment_id: ID of the segment to remove
                force: If False, will not remove segments with allocated blocks

        Returns:
                True if segment was removed, False otherwise
        """
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
        """
        Get all traces of a specific operation across all segments and blocks.

        Searches through all segments and blocks to find traces matching
        the specified operation type.

        Args:
                operation: Operation type to search for

        Returns:
                List of trace dictionaries sorted by timestamp
        """
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
        """
        Get all traces from all segments and blocks, sorted by timestamp.

        Collects every trace from every segment and block in the pool,
        providing a complete timeline of memory operations.

        Returns:
                List of all trace dictionaries sorted chronologically
        """
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
        """
        Analyze memory allocation patterns based on traces.

        Processes all trace information to provide insights into memory
        usage patterns, operation frequencies, and timing behavior.

        Returns:
                Dictionary containing analysis results and statistics
        """
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
        """
        Get overall memory summary across all segments.

        Provides high-level statistics about memory usage, fragmentation,
        and utilization across the entire pool.

        Returns:
                Dictionary containing summary statistics
        """
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
        """
        Print detailed memory status for debugging.

        Outputs a comprehensive report of the current memory state including
        segment information, block details, and utilization statistics.
        """
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
