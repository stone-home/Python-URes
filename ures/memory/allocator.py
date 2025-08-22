from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
import random
import time
from dataclasses import dataclass

# Import from the existing blocks.py file
from ures.memory import BlockPool, MemoryBlock, Segment, TraceInfo


class AllocationStrategy(Enum):
    """Enumeration of allocation strategies"""

    FIRST_FIT = "first_fit"
    BEST_FIT = "best_fit"
    WORST_FIT = "worst_fit"
    NEXT_FIT = "next_fit"
    BUDDY_SYSTEM = "buddy_system"


@dataclass
class AllocationRequest:
    """Represents a memory allocation request"""

    size: int
    alignment: int = 1
    stream: Optional[int] = None
    priority: int = 0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AllocationResult:
    """Represents the result of an allocation attempt"""

    success: bool
    block: Optional[MemoryBlock] = None
    address: Optional[int] = None
    actual_size: Optional[int] = None
    allocation_time_ns: Optional[int] = None
    error_message: Optional[str] = None
    strategy_info: Optional[Dict[str, Any]] = None


@dataclass
class FreeRequest:
    """Represents a memory deallocation request"""

    address: int
    expected_size: Optional[int] = None  # For validation
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class FreeResult:
    """Represents the result of a deallocation attempt"""

    success: bool
    address: Optional[int] = None
    freed_size: Optional[int] = None
    free_time_ns: Optional[int] = None
    coalesced: bool = False
    coalesced_size: Optional[int] = None
    error_message: Optional[str] = None
    strategy_info: Optional[Dict[str, Any]] = None


class MemoryAllocator(ABC):
    """Abstract base class for memory allocation algorithms"""

    def __init__(self, name: str):
        self.name = name
        self.allocation_count = 0
        self.free_count = 0
        self.total_allocated = 0
        self.total_freed = 0
        self.allocation_times = []
        self.free_times = []
        self.allocated_blocks: Dict[int, MemoryBlock] = (
            {}
        )  # Track allocated blocks by address

    @abstractmethod
    def allocate(self, pool: BlockPool, request: AllocationRequest) -> AllocationResult:
        """Allocate memory according to the algorithm's strategy"""
        pass

    @abstractmethod
    def free(self, pool: BlockPool, request: FreeRequest) -> FreeResult:
        """Free memory according to the algorithm's strategy"""
        pass

    @abstractmethod
    def can_allocate(self, pool: BlockPool, request: AllocationRequest) -> bool:
        """Check if allocation is possible without actually allocating"""
        pass

    def can_free(self, pool: BlockPool, request: FreeRequest) -> bool:
        """Check if deallocation is possible"""
        return request.address in self.allocated_blocks

    def get_statistics(self) -> Dict[str, Any]:
        """Get allocator statistics"""
        avg_alloc_time = (
            sum(self.allocation_times) / len(self.allocation_times)
            if self.allocation_times
            else 0
        )
        avg_free_time = (
            sum(self.free_times) / len(self.free_times) if self.free_times else 0
        )

        return {
            "name": self.name,
            "allocation_count": self.allocation_count,
            "free_count": self.free_count,
            "total_allocated": self.total_allocated,
            "total_freed": self.total_freed,
            "currently_allocated": self.total_allocated - self.total_freed,
            "active_blocks": len(self.allocated_blocks),
            "average_allocation_time_ns": avg_alloc_time,
            "average_free_time_ns": avg_free_time,
            "allocation_times": self.allocation_times.copy(),
            "free_times": self.free_times.copy(),
        }


class FirstFitAllocator(MemoryAllocator):
    """First-fit allocation algorithm"""

    def __init__(self):
        super().__init__("First Fit")

    def allocate(self, pool: BlockPool, request: AllocationRequest) -> AllocationResult:
        start_time = time.time_ns()

        # For first fit, just iterate through all blocks in order
        search_steps = 0

        for block in pool.blocks:
            search_steps += 1
            if (
                not block.value.is_allocated()
                and block.value.size >= request.size
                and (request.stream is None or block.stream == request.stream)
            ):
                # Found a suitable block
                allocated_block = block.splice(request.size)
                allocated_block.request_alloc(time_ns=start_time)

                # Track the allocated block
                self.allocated_blocks[allocated_block.addr] = allocated_block

                self.allocation_count += 1
                self.total_allocated += request.size
                allocation_time = time.time_ns() - start_time
                self.allocation_times.append(allocation_time)

                return AllocationResult(
                    success=True,
                    block=allocated_block,
                    address=allocated_block.addr,
                    actual_size=allocated_block.value.size,
                    allocation_time_ns=allocation_time,
                    strategy_info={
                        "algorithm": "first_fit",
                        "search_steps": search_steps,
                    },
                )

        allocation_time = time.time_ns() - start_time
        self.allocation_times.append(allocation_time)

        return AllocationResult(
            success=False,
            allocation_time_ns=allocation_time,
            error_message=f"No suitable block found for size {request.size}",
            strategy_info={"algorithm": "first_fit", "search_steps": search_steps},
        )

    def free(self, pool: BlockPool, request: FreeRequest) -> FreeResult:
        start_time = time.time_ns()

        # Check if we have this block
        if request.address not in self.allocated_blocks:
            free_time = time.time_ns() - start_time
            self.free_times.append(free_time)
            return FreeResult(
                success=False,
                address=request.address,
                free_time_ns=free_time,
                error_message=f"Block at address {hex(request.address)} not found or not allocated by this allocator",
            )

        block = self.allocated_blocks[request.address]

        # Validate expected size if provided
        if (
            request.expected_size is not None
            and block.value.size != request.expected_size
        ):
            free_time = time.time_ns() - start_time
            self.free_times.append(free_time)
            return FreeResult(
                success=False,
                address=request.address,
                free_time_ns=free_time,
                error_message=f"Size mismatch: expected {request.expected_size}, actual {block.value.size}",
            )

        # Free the block
        original_size = block.value.size
        block.request_free(time_ns=start_time)
        block.complete_free()

        # Try to coalesce with adjacent free blocks
        coalesced_block = block.coalesce()
        coalesced = coalesced_block.value.size > original_size

        # Remove from our tracking
        del self.allocated_blocks[request.address]

        self.free_count += 1
        self.total_freed += original_size
        free_time = time.time_ns() - start_time
        self.free_times.append(free_time)

        return FreeResult(
            success=True,
            address=request.address,
            freed_size=original_size,
            free_time_ns=free_time,
            coalesced=coalesced,
            coalesced_size=coalesced_block.value.size if coalesced else None,
            strategy_info={"algorithm": "first_fit", "coalescing": "immediate"},
        )

    def can_allocate(self, pool: BlockPool, request: AllocationRequest) -> bool:
        # Simple linear search for feasibility check
        for block in pool.blocks:
            if (
                not block.value.is_allocated()
                and block.value.size >= request.size
                and (request.stream is None or block.stream == request.stream)
            ):
                return True
        return False


class BestFitAllocator(MemoryAllocator):
    """Best-fit allocation algorithm"""

    def __init__(self):
        super().__init__("Best Fit")

    def allocate(self, pool: BlockPool, request: AllocationRequest) -> AllocationResult:
        start_time = time.time_ns()

        best_block = None
        best_size = float("inf")
        search_steps = 0

        # Search all free blocks for the best fit
        for block in pool.blocks:
            search_steps += 1
            if (
                not block.value.is_allocated()
                and block.value.size >= request.size
                and (request.stream is None or block.stream == request.stream)
                and block.value.size < best_size
            ):
                best_block = block
                best_size = block.value.size

        if best_block:
            allocated_block = best_block.splice(request.size)
            allocated_block.request_alloc(time_ns=start_time)

            # Track the allocated block
            self.allocated_blocks[allocated_block.addr] = allocated_block

            self.allocation_count += 1
            self.total_allocated += request.size
            allocation_time = time.time_ns() - start_time
            self.allocation_times.append(allocation_time)

            return AllocationResult(
                success=True,
                block=allocated_block,
                address=allocated_block.addr,
                actual_size=allocated_block.value.size,
                allocation_time_ns=allocation_time,
                strategy_info={
                    "algorithm": "best_fit",
                    "search_steps": search_steps,
                    "waste": best_size - request.size,
                },
            )

        allocation_time = time.time_ns() - start_time
        self.allocation_times.append(allocation_time)

        return AllocationResult(
            success=False,
            allocation_time_ns=allocation_time,
            error_message=f"No suitable block found for size {request.size}",
            strategy_info={"algorithm": "best_fit", "search_steps": search_steps},
        )

    def free(self, pool: BlockPool, request: FreeRequest) -> FreeResult:
        start_time = time.time_ns()

        # Check if we have this block
        if request.address not in self.allocated_blocks:
            free_time = time.time_ns() - start_time
            self.free_times.append(free_time)
            return FreeResult(
                success=False,
                address=request.address,
                free_time_ns=free_time,
                error_message=f"Block at address {hex(request.address)} not found or not allocated by this allocator",
            )

        block = self.allocated_blocks[request.address]

        # Validate expected size if provided
        if (
            request.expected_size is not None
            and block.value.size != request.expected_size
        ):
            free_time = time.time_ns() - start_time
            self.free_times.append(free_time)
            return FreeResult(
                success=False,
                address=request.address,
                free_time_ns=free_time,
                error_message=f"Size mismatch: expected {request.expected_size}, actual {block.value.size}",
            )

        # Free the block
        original_size = block.value.size
        block.request_free(time_ns=start_time)
        block.complete_free()

        # Try to coalesce with adjacent free blocks
        coalesced_block = block.coalesce()
        coalesced = coalesced_block.value.size > original_size

        # Remove from our tracking
        del self.allocated_blocks[request.address]

        self.free_count += 1
        self.total_freed += original_size
        free_time = time.time_ns() - start_time
        self.free_times.append(free_time)

        return FreeResult(
            success=True,
            address=request.address,
            freed_size=original_size,
            free_time_ns=free_time,
            coalesced=coalesced,
            coalesced_size=coalesced_block.value.size if coalesced else None,
            strategy_info={"algorithm": "best_fit", "coalescing": "immediate"},
        )

    def can_allocate(self, pool: BlockPool, request: AllocationRequest) -> bool:
        for block in pool.blocks:
            if (
                not block.value.is_allocated()
                and block.value.size >= request.size
                and (request.stream is None or block.stream == request.stream)
            ):
                return True
        return False


class WorstFitAllocator(MemoryAllocator):
    """Worst-fit allocation algorithm"""

    def __init__(self):
        super().__init__("Worst Fit")

    def allocate(self, pool: BlockPool, request: AllocationRequest) -> AllocationResult:
        start_time = time.time_ns()

        worst_block = None
        worst_size = 0
        search_steps = 0

        # Search all free blocks for the worst fit (largest)
        for block in pool.blocks:
            search_steps += 1
            if (
                not block.value.is_allocated()
                and block.value.size >= request.size
                and (request.stream is None or block.stream == request.stream)
                and block.value.size > worst_size
            ):
                worst_block = block
                worst_size = block.value.size

        if worst_block:
            allocated_block = worst_block.splice(request.size)
            allocated_block.request_alloc(time_ns=start_time)

            # Track the allocated block
            self.allocated_blocks[allocated_block.addr] = allocated_block

            self.allocation_count += 1
            self.total_allocated += request.size
            allocation_time = time.time_ns() - start_time
            self.allocation_times.append(allocation_time)

            return AllocationResult(
                success=True,
                block=allocated_block,
                address=allocated_block.addr,
                actual_size=allocated_block.value.size,
                allocation_time_ns=allocation_time,
                strategy_info={
                    "algorithm": "worst_fit",
                    "search_steps": search_steps,
                    "remaining_in_block": worst_size - request.size,
                },
            )

        allocation_time = time.time_ns() - start_time
        self.allocation_times.append(allocation_time)

        return AllocationResult(
            success=False,
            allocation_time_ns=allocation_time,
            error_message=f"No suitable block found for size {request.size}",
            strategy_info={"algorithm": "worst_fit", "search_steps": search_steps},
        )

    def free(self, pool: BlockPool, request: FreeRequest) -> FreeResult:
        start_time = time.time_ns()

        # Check if we have this block
        if request.address not in self.allocated_blocks:
            free_time = time.time_ns() - start_time
            self.free_times.append(free_time)
            return FreeResult(
                success=False,
                address=request.address,
                free_time_ns=free_time,
                error_message=f"Block at address {hex(request.address)} not found or not allocated by this allocator",
            )

        block = self.allocated_blocks[request.address]

        # Validate expected size if provided
        if (
            request.expected_size is not None
            and block.value.size != request.expected_size
        ):
            free_time = time.time_ns() - start_time
            self.free_times.append(free_time)
            return FreeResult(
                success=False,
                address=request.address,
                free_time_ns=free_time,
                error_message=f"Size mismatch: expected {request.expected_size}, actual {block.value.size}",
            )

        # Free the block
        original_size = block.value.size
        block.request_free(time_ns=start_time)
        block.complete_free()

        # Try to coalesce with adjacent free blocks
        coalesced_block = block.coalesce()
        coalesced = coalesced_block.value.size > original_size

        # Remove from our tracking
        del self.allocated_blocks[request.address]

        self.free_count += 1
        self.total_freed += original_size
        free_time = time.time_ns() - start_time
        self.free_times.append(free_time)

        return FreeResult(
            success=True,
            address=request.address,
            freed_size=original_size,
            free_time_ns=free_time,
            coalesced=coalesced,
            coalesced_size=coalesced_block.value.size if coalesced else None,
            strategy_info={"algorithm": "worst_fit", "coalescing": "immediate"},
        )

    def can_allocate(self, pool: BlockPool, request: AllocationRequest) -> bool:
        for block in pool.blocks:
            if (
                not block.value.is_allocated()
                and block.value.size >= request.size
                and (request.stream is None or block.stream == request.stream)
            ):
                return True
        return False


class NextFitAllocator(MemoryAllocator):
    """Next-fit allocation algorithm"""

    def __init__(self):
        super().__init__("Next Fit")
        self.last_allocated_block = None  # Remember where we last allocated

    def allocate(self, pool: BlockPool, request: AllocationRequest) -> AllocationResult:
        start_time = time.time_ns()
        search_steps = 0

        # Convert blocks to list for easier indexing
        blocks_list = list(pool.blocks)
        if not blocks_list:
            return AllocationResult(
                success=False,
                allocation_time_ns=time.time_ns() - start_time,
                error_message="No blocks available",
            )

        # Find starting position
        start_idx = 0
        if self.last_allocated_block:
            # Try to find the last allocated block in current pool
            for i, block in enumerate(blocks_list):
                if block.addr >= self.last_allocated_block.addr:
                    start_idx = i
                    break

        # Search from last position to end, then wrap around
        for attempt in range(2):  # Two attempts: from last pos to end, then from start
            start = start_idx if attempt == 0 else 0
            end = len(blocks_list) if attempt == 0 else start_idx

            for i in range(start, end):
                block = blocks_list[i]
                search_steps += 1

                if (
                    not block.value.is_allocated()
                    and block.value.size >= request.size
                    and self._stream_matches(block.stream, request.stream)
                ):
                    # Found a suitable block
                    allocated_block = block.splice(request.size)
                    allocated_block.request_alloc(time_ns=start_time)

                    # Track the allocated block
                    self.allocated_blocks[allocated_block.addr] = allocated_block
                    self.last_allocated_block = (
                        allocated_block  # Remember for next time
                    )

                    self.allocation_count += 1
                    self.total_allocated += request.size
                    allocation_time = time.time_ns() - start_time
                    self.allocation_times.append(allocation_time)

                    return AllocationResult(
                        success=True,
                        block=allocated_block,
                        address=allocated_block.addr,
                        actual_size=allocated_block.value.size,
                        allocation_time_ns=allocation_time,
                        strategy_info={
                            "algorithm": "next_fit",
                            "search_steps": search_steps,
                            "start_position": start_idx,
                        },
                    )

        allocation_time = time.time_ns() - start_time
        self.allocation_times.append(allocation_time)

        return AllocationResult(
            success=False,
            allocation_time_ns=allocation_time,
            error_message=f"No suitable block found for size {request.size}",
            strategy_info={"algorithm": "next_fit", "search_steps": search_steps},
        )

    def free(self, pool: BlockPool, request: FreeRequest) -> FreeResult:
        start_time = time.time_ns()

        # Check if we have this block
        if request.address not in self.allocated_blocks:
            free_time = time.time_ns() - start_time
            self.free_times.append(free_time)
            return FreeResult(
                success=False,
                address=request.address,
                free_time_ns=free_time,
                error_message=f"Block at address {hex(request.address)} not found or not allocated by this allocator",
            )

        block = self.allocated_blocks[request.address]

        # Validate expected size if provided
        if (
            request.expected_size is not None
            and block.value.size != request.expected_size
        ):
            free_time = time.time_ns() - start_time
            self.free_times.append(free_time)
            return FreeResult(
                success=False,
                address=request.address,
                free_time_ns=free_time,
                error_message=f"Size mismatch: expected {request.expected_size}, actual {block.value.size}",
            )

        # Free the block
        original_size = block.value.size
        block.request_free(time_ns=start_time)
        block.complete_free()

        # Try to coalesce with adjacent free blocks
        coalesced_block = block.coalesce()
        coalesced = coalesced_block.value.size > original_size

        # Reset last allocated position if we freed that block
        if (
            self.last_allocated_block
            and self.last_allocated_block.addr == request.address
        ):
            self.last_allocated_block = None

        # Remove from our tracking
        del self.allocated_blocks[request.address]

        self.free_count += 1
        self.total_freed += original_size
        free_time = time.time_ns() - start_time
        self.free_times.append(free_time)

        return FreeResult(
            success=True,
            address=request.address,
            freed_size=original_size,
            free_time_ns=free_time,
            coalesced=coalesced,
            coalesced_size=coalesced_block.value.size if coalesced else None,
            strategy_info={"algorithm": "next_fit", "coalescing": "immediate"},
        )

    def can_allocate(self, pool: BlockPool, request: AllocationRequest) -> bool:
        # Use the parent's implementation which already has stream matching logic
        for block in pool.blocks:
            if (
                not block.value.is_allocated()
                and block.value.size >= request.size
                and self._stream_matches(block.stream, request.stream)
            ):
                return True
        return False

    def _stream_matches(
        self, block_stream: Optional[int], request_stream: Optional[int]
    ) -> bool:
        """Helper method to safely compare streams"""
        if request_stream is None:
            return True  # Request doesn't care about stream
        return block_stream == request_stream


class BuddySystemAllocator(MemoryAllocator):
    """Buddy system allocation algorithm"""

    def __init__(self):
        super().__init__("Buddy System")
        self.min_block_size = 64  # Minimum block size (can be configured)

    def _round_up_to_power_of_2(self, size: int) -> int:
        """Round up size to next power of 2"""
        if size <= 0:
            return self.min_block_size

        # Handle sizes smaller than minimum
        if size < self.min_block_size:
            return self.min_block_size

        # Round up to next power of 2
        power = 1
        while power < size:
            power <<= 1
        return power

    def _is_power_of_2(self, n: int) -> bool:
        """Check if number is power of 2"""
        return n > 0 and (n & (n - 1)) == 0

    def _find_buddy_address(self, addr: int, size: int) -> int:
        """Find the buddy address for a given block"""
        return addr ^ size

    def _can_merge_with_buddy(
        self, pool: BlockPool, block_addr: int, block_size: int
    ) -> Optional[MemoryBlock]:
        """Check if block can merge with its buddy"""
        buddy_addr = self._find_buddy_address(block_addr, block_size)

        # Find the buddy block
        for block in pool.blocks:
            if (
                block.addr == buddy_addr
                and block.value.size == block_size
                and not block.value.is_allocated()
            ):
                return block
        return None

    def allocate(self, pool: BlockPool, request: AllocationRequest) -> AllocationResult:
        start_time = time.time_ns()

        # Round up request size to power of 2
        buddy_size = self._round_up_to_power_of_2(request.size)
        search_steps = 0

        # Find the smallest suitable block
        best_block = None
        best_size = float("inf")

        for block in pool.blocks:
            search_steps += 1
            if (
                not block.value.is_allocated()
                and self._is_power_of_2(block.value.size)
                and block.value.size >= buddy_size
                and block.value.size < best_size
                and self._stream_matches(block.stream, request.stream)
            ):
                best_block = block
                best_size = block.value.size

        if not best_block:
            allocation_time = time.time_ns() - start_time
            self.allocation_times.append(allocation_time)
            return AllocationResult(
                success=False,
                allocation_time_ns=allocation_time,
                error_message=f"No suitable buddy block found for size {buddy_size}",
                strategy_info={
                    "algorithm": "buddy_system",
                    "search_steps": search_steps,
                },
            )

        # Split the block down to required size
        current_block = best_block
        splits_performed = 0

        while current_block.value.size > buddy_size:
            # Split the block in half
            half_size = current_block.value.size // 2
            split_block = current_block.splice(half_size)
            splits_performed += 1

            # Continue with the first half
            current_block = split_block

        # Allocate the final block
        allocated_block = current_block
        allocated_block.request_alloc(time_ns=start_time)

        # Track the allocated block
        self.allocated_blocks[allocated_block.addr] = allocated_block

        self.allocation_count += 1
        self.total_allocated += buddy_size
        allocation_time = time.time_ns() - start_time
        self.allocation_times.append(allocation_time)

        return AllocationResult(
            success=True,
            block=allocated_block,
            address=allocated_block.addr,
            actual_size=allocated_block.value.size,
            allocation_time_ns=allocation_time,
            strategy_info={
                "algorithm": "buddy_system",
                "search_steps": search_steps,
                "buddy_size": buddy_size,
                "original_request_size": request.size,
                "internal_fragmentation": buddy_size - request.size,
                "splits_performed": splits_performed,
            },
        )

    def free(self, pool: BlockPool, request: FreeRequest) -> FreeResult:
        start_time = time.time_ns()

        # Check if we have this block
        if request.address not in self.allocated_blocks:
            free_time = time.time_ns() - start_time
            self.free_times.append(free_time)
            return FreeResult(
                success=False,
                address=request.address,
                free_time_ns=free_time,
                error_message=f"Block at address {hex(request.address)} not found or not allocated by this allocator",
            )

        block = self.allocated_blocks[request.address]
        original_size = block.value.size

        # Free the block
        block.request_free(time_ns=start_time)
        block.complete_free()

        # Buddy system coalescing: try to merge with buddy repeatedly
        current_block = block
        merges_performed = 0

        while True:
            buddy_block = self._can_merge_with_buddy(
                pool, current_block.addr, current_block.value.size
            )
            if not buddy_block:
                break

            # Merge with buddy
            if current_block.addr < buddy_block.addr:
                # Current block is the left buddy
                current_block.value.size += buddy_block.value.size
                buddy_block.remove()
                if buddy_block in pool.blocks:
                    pool.blocks.remove(buddy_block)
            else:
                # Buddy block is the left buddy
                buddy_block.value.size += current_block.value.size
                current_block.remove()
                if current_block in pool.blocks:
                    pool.blocks.remove(current_block)
                current_block = buddy_block

            merges_performed += 1

        # Remove from our tracking
        del self.allocated_blocks[request.address]

        self.free_count += 1
        self.total_freed += original_size
        free_time = time.time_ns() - start_time
        self.free_times.append(free_time)

        return FreeResult(
            success=True,
            address=request.address,
            freed_size=original_size,
            free_time_ns=free_time,
            coalesced=merges_performed > 0,
            coalesced_size=current_block.value.size if merges_performed > 0 else None,
            strategy_info={
                "algorithm": "buddy_system",
                "coalescing": "buddy_merging",
                "merges_performed": merges_performed,
            },
        )

    def can_allocate(self, pool: BlockPool, request: AllocationRequest) -> bool:
        buddy_size = self._round_up_to_power_of_2(request.size)

        for block in pool.blocks:
            if (
                not block.value.is_allocated()
                and block.value.size >= buddy_size
                and self._stream_matches(block.stream, request.stream)
            ):
                return True
        return False

    def _stream_matches(
        self, block_stream: Optional[int], request_stream: Optional[int]
    ) -> bool:
        """Helper method to safely compare streams"""
        if request_stream is None:
            return True  # Request doesn't care about stream
        return block_stream == request_stream


class DeviceMemorySimulator:
    """Simulates a device with configurable memory allocation algorithms"""

    def __init__(
        self, device_id: int, total_memory: int, base_address: int = 0x10000000
    ):
        self.device_id = device_id
        self.total_memory = total_memory
        self.base_address = base_address
        self.pool = BlockPool()
        self.allocators: Dict[str, MemoryAllocator] = {}
        self.current_allocator: Optional[MemoryAllocator] = None
        self.allocation_history: List[Tuple[AllocationRequest, AllocationResult]] = []
        self.free_history: List[Tuple[int, int]] = []  # (address, time_ns)

        # Create initial memory segment
        self.main_segment = self.pool.create_segment(
            start_addr=base_address, size=total_memory, device=device_id
        )

        # Register default allocators
        self.register_allocator(FirstFitAllocator())
        self.register_allocator(BestFitAllocator())
        self.register_allocator(WorstFitAllocator())
        self.register_allocator(NextFitAllocator())
        self.register_allocator(BuddySystemAllocator())

        # Set default allocator
        self.set_allocator("First Fit")

    def register_allocator(self, allocator: MemoryAllocator):
        """Register a new allocation algorithm"""
        self.allocators[allocator.name] = allocator

    def set_allocator(self, allocator_name: str) -> bool:
        """Set the active allocation algorithm"""
        if allocator_name in self.allocators:
            self.current_allocator = self.allocators[allocator_name]
            return True
        return False

    def get_available_allocators(self) -> List[str]:
        """Get list of available allocator names"""
        return list(self.allocators.keys())

    def allocate(
        self,
        size: int,
        alignment: int = 1,
        stream: Optional[int] = None,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AllocationResult:
        """Allocate memory using the current algorithm"""
        if not self.current_allocator:
            return AllocationResult(success=False, error_message="No allocator set")

        request = AllocationRequest(
            size=size,
            alignment=alignment,
            stream=stream,
            priority=priority,
            metadata=metadata,
        )

        result = self.current_allocator.allocate(self.pool, request)
        self.allocation_history.append((request, result))

        return result

    def free(self, address: int, expected_size: Optional[int] = None) -> FreeResult:
        """Free memory at the given address using the current allocator"""
        if not self.current_allocator:
            return FreeResult(
                success=False, address=address, error_message="No allocator set"
            )

        request = FreeRequest(address=address, expected_size=expected_size)

        result = self.current_allocator.free(self.pool, request)
        self.free_history.append((address, time.time_ns()))

        return result

    def can_allocate(self, size: int, stream: Optional[int] = None) -> bool:
        """Check if allocation is possible without actually allocating"""
        if not self.current_allocator:
            return False

        request = AllocationRequest(size=size, stream=stream)
        return self.current_allocator.can_allocate(self.pool, request)

    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory status information"""
        summary = self.pool.get_memory_summary()

        return {
            "device_id": self.device_id,
            "total_memory": self.total_memory,
            "base_address": self.base_address,
            "base_address_hex": hex(self.base_address),
            "current_allocator": (
                self.current_allocator.name if self.current_allocator else None
            ),
            "memory_summary": summary,
            "allocation_count": len(self.allocation_history),
            "free_count": len(self.free_history),
            "segments": self.pool.list_all_segments(),
        }

    def get_allocator_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all allocators"""
        stats = {}
        for name, allocator in self.allocators.items():
            stats[name] = allocator.get_statistics()
        return stats

    def reset_device(self):
        """Reset device to initial state"""
        # Clear all data
        self.pool = BlockPool()
        self.allocation_history.clear()
        self.free_history.clear()

        # Reset allocator statistics
        for allocator in self.allocators.values():
            allocator.allocation_count = 0
            allocator.free_count = 0
            allocator.total_allocated = 0
            allocator.total_freed = 0
            allocator.allocation_times.clear()
            allocator.free_times.clear()
            allocator.allocated_blocks.clear()

        # Recreate main segment
        self.main_segment = self.pool.create_segment(
            start_addr=self.base_address, size=self.total_memory, device=self.device_id
        )

    def simulate_workload(
        self,
        num_operations: int = 100,
        size_range: Tuple[int, int] = (64, 4096),
        free_probability: float = 0.3,
    ) -> Dict[str, Any]:
        """Simulate a random workload"""
        operations = []
        allocated_blocks = []

        for i in range(num_operations):
            if allocated_blocks and random.random() < free_probability:
                # Free a random block
                block_to_free = random.choice(allocated_blocks)
                result = self.free(block_to_free)
                operations.append(
                    {
                        "operation": "free",
                        "address": block_to_free,
                        "success": result.success,
                    }
                )
                if result.success:
                    allocated_blocks.remove(block_to_free)
            else:
                # Allocate a new block
                size = random.randint(*size_range)
                result = self.allocate(size)
                operations.append(
                    {
                        "operation": "allocate",
                        "size": size,
                        "success": result.success,
                        "address": result.address,
                    }
                )
                if result.success:
                    allocated_blocks.append(result.address)

        return {
            "operations": operations,
            "final_memory_info": self.get_memory_info(),
            "allocator_stats": self.get_allocator_statistics(),
        }

    def print_status(self):
        """Print current device status"""
        print(f"=== Device {self.device_id} Memory Status ===")
        info = self.get_memory_info()
        summary = info["memory_summary"]

        print(f"Total Memory: {self.total_memory} bytes")
        print(f"Base Address: {info['base_address_hex']}")
        print(f"Current Allocator: {info['current_allocator']}")
        print(
            f"Allocated: {summary['total_allocated_bytes']} bytes ({summary['overall_utilization']:.1%})"
        )
        print(f"Free: {summary['total_free_bytes']} bytes")
        print(f"Fragmentation: {summary['average_fragmentation']:.1%}")
        print(f"Total Allocations: {info['allocation_count']}")
        print(f"Total Frees: {info['free_count']}")

        self.pool.print_memory_status()
