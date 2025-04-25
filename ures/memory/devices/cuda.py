import cupy
import pynvml
import bisect
import uuid  # Optional for alternative handle generation
from .interface import DeviceInterface


class CUDAAllocator(DeviceInterface):
    """
    A memory allocator that interacts with a real NVIDIA GPU using CuPy and pynvml.
    Implements the DeviceInterface standard.
    """

    def __init__(self, device_index: int = 0):
        """
        Initializes the allocator for a specific GPU device.

        Args:
            device_index: The index of the GPU device to use (default: 0).
        """
        self.device_index = device_index
        self.allocations = {}
        self.pynvml_initialized = False
        self.nvml_handle = None

        try:
            self.cupy_device = cupy.cuda.Device(device_index)
            # Activate the device context for initialization and future calls if needed
            self.cupy_device.use()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CuPy device {device_index}: {e}")

        try:
            pynvml.nvmlInit()
            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self.pynvml_initialized = True
        except pynvml.NVMLError as error:
            print(
                f"Warning: pynvml initialization failed for device {device_index}: {error}. "
                "Device name and potentially ID might be unavailable."
            )

    def device_id(self) -> str:
        """Returns the PCI Bus ID of the device."""
        try:
            # Use PCI Bus ID from CuPy as the primary identifier
            return self.cupy_device.pci_bus_id
        except Exception as e:
            print(f"Warning: Failed to get PCI Bus ID via CuPy: {e}")
            # Fallback or alternative ID generation if needed
            return f"GPU_{self.device_index}_ID_Unavailable"

    def name(self) -> str:
        """Returns the name of the GPU device."""
        if self.pynvml_initialized and self.nvml_handle:
            try:
                name_bytes = pynvml.nvmlDeviceGetName(self.nvml_handle)
                return name_bytes.decode("utf-8")
            except pynvml.NVMLError as error:
                print(f"Warning: Failed to get device name via pynvml: {error}")
        return f"Unknown Device (Index {self.device_index})"  # Fallback name

    def total_memory(self) -> int:
        """Returns the total memory capacity in bytes."""
        try:
            # mem_info returns (free, total)
            return self.cupy_device.mem_info[1]
        except Exception as e:
            raise RuntimeError(
                f"Failed to get total memory for device {self.device_index}: {e}"
            )

    def available_memory(self) -> int:
        """
        Returns the currently available physical memory in bytes.
        Note: This might not reflect the memory available in CuPy's pool.
        """
        try:
            # mem_info returns (free, total)
            return self.cupy_device.mem_info
        except Exception as e:
            raise RuntimeError(
                f"Failed to get available memory for device {self.device_index}: {e}"
            )

    def malloc(self, size: int) -> str:
        """
        Allocates memory on the GPU.

        Args:
            size: The number of bytes to allocate.

        Returns:
            A string handle representing the allocation.

        Raises:
            ValueError: If size is not positive.
            MemoryError: If allocation fails due to insufficient memory or other CUDA errors.
            RuntimeError: For unexpected errors or handle collisions.
        """
        if size <= 0:
            raise ValueError("Allocation size must be positive")

        memory_pointer = None
        try:
            # Ensure allocation happens on the correct device context
            with self.cupy_device:
                memory_pointer = cupy.cuda.alloc(size)
        except cupy.cuda.driver.CUDADriverError as e:
            # Check e.status for specific CUDA error codes if needed, e.g., cudaErrorMemoryAllocation
            raise MemoryError(
                f"GPU memory allocation of {size} bytes failed on device {self.device_index}. Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during allocation: {e}")

        if memory_pointer is None:
            raise MemoryError(
                f"GPU memory allocation of {size} bytes failed (returned None)."
            )

        # Handle Generation (using address as string)
        ptr_int = memory_pointer.ptr
        handle_str = str(ptr_int)

        # Handle collision check (unlikely but good practice if using address)
        if handle_str in self.allocations:
            raise RuntimeError(
                f"Handle collision detected for address {ptr_int}. Consider using UUID handles."
            )

        # Track Allocation: Store the MemoryPointer object itself
        self.allocations[handle_str] = {
            "ptr": ptr_int,
            "size": size,
            "mem_obj": memory_pointer,
        }
        return handle_str

    def free(self, handle: str) -> None:
        """
        Frees a previously allocated memory block identified by the handle.
        This returns the memory to the CuPy memory pool.

        Args:
            handle: The string handle returned by malloc.

        Raises:
            ValueError: If the handle is invalid.
        """
        if handle not in self.allocations:
            raise ValueError(f"Invalid memory handle provided to free: {handle}")

        # Retrieve the allocation info to ensure the object reference exists
        alloc_info = self.allocations[handle]
        mem_obj = alloc_info["mem_obj"]  # Reference to the MemoryPointer

        # Remove the allocator's reference. GC will eventually trigger pool return.
        del self.allocations[handle]

        # The 'mem_obj' variable holding the MemoryPointer will go out of scope
        # if this was the last reference, triggering GC and pool management.

    def shutdown(self) -> None:
        """Shuts down pynvml if it was initialized."""
        if self.pynvml_initialized:
            try:
                pynvml.nvmlShutdown()
                self.pynvml_initialized = False
            except pynvml.NVMLError as error:
                print(f"Warning: pynvml shutdown failed: {error}")

    def __del__(self):
        """Ensures shutdown is called when the object is destroyed."""
        self.shutdown()


class SimCUDAAllocator:
    """
    Simulates a GPU memory allocator using the First Fit algorithm.
    Implements the DeviceInterface standard without hardware interaction.
    """

    def __init__(self, device_id: str, name: str, total_memory: int):
        """
        Initializes the simulated allocator.

        Args:
            device_id: A unique string identifier for the simulated device.
            name: A human-readable name for the simulated device.
            total_memory: The total memory capacity in bytes for the simulation.

        Raises:
            ValueError: If total_memory is not positive.
        """
        if total_memory <= 0:
            raise ValueError("Total memory must be positive")

        self._sim_device_id = device_id
        self._sim_name = name
        self._sim_total_memory = total_memory

        # Stores allocated blocks: {handle: {'address': int, 'size': int}}
        self.allocated_blocks = {}
        # Stores free blocks as list of (start_address, size) tuples, sorted by start_address
        self.free_blocks = [(0, total_memory)]

    def device_id(self) -> str:
        """Returns the configured device ID."""
        return self._sim_device_id

    def name(self) -> str:
        """Returns the configured device name."""
        return self._sim_name

    def total_memory(self) -> int:
        """Returns the configured total memory."""
        return self._sim_total_memory

    def available_memory(self) -> int:
        """Returns the total size of all free blocks."""
        return sum(size for start, size in self.free_blocks)

    def malloc(self, size: int) -> str:
        """
        Allocates simulated memory using the First Fit algorithm.

        Args:
            size: The number of bytes to allocate.

        Returns:
            A string handle representing the allocation (based on address).

        Raises:
            ValueError: If size is not positive.
            MemoryError: If no suitable free block is found.
            RuntimeError: For internal errors like handle collisions.
        """
        if size <= 0:
            raise ValueError("Allocation size must be positive")

        best_fit_index = -1
        # Find the first block that fits (First Fit)
        for i, block in enumerate(self.free_blocks):
            free_start, free_size = block
            if free_size >= size:
                best_fit_index = i
                break  # Found the first fit

        if best_fit_index == -1:
            raise MemoryError(
                f"Simulated allocation failed: Cannot allocate {size} bytes. "
                f"Available: {self.available_memory()} bytes in {len(self.free_blocks)} fragments."
            )

        # Get the chosen block details and remove it from free list
        chosen_block = self.free_blocks.pop(best_fit_index)
        alloc_address, block_size = chosen_block

        # Generate handle (using address as string)
        handle = str(alloc_address)
        if handle in self.allocated_blocks:
            raise RuntimeError(
                f"Internal error: Handle collision for address {alloc_address}"
            )

        # Record allocation
        self.allocated_blocks[handle] = {"address": alloc_address, "size": size}

        # Add remaining fragment back to free list if necessary
        remaining_size = block_size - size
        if remaining_size > 0:
            remaining_start = alloc_address + size
            remaining_block = (remaining_start, remaining_size)
            # Insert back while maintaining sorted order
            bisect.insort_left(self.free_blocks, remaining_block)

        return handle

    def free(self, handle: str) -> None:
        """
        Frees a simulated memory allocation and coalesces free blocks.

        Args:
            handle: The string handle returned by malloc.

        Raises:
            ValueError: If the handle is invalid.
        """
        if handle not in self.allocated_blocks:
            raise ValueError(f"Invalid memory handle provided to free: {handle}")

        alloc_info = self.allocated_blocks.pop(handle)  # Remove and get info
        freed_address = alloc_info["address"]
        freed_size = alloc_info["size"]

        new_free_block = (freed_address, freed_size)

        # Insert the freed block into the sorted free list
        insert_pos = bisect.bisect_left(self.free_blocks, new_free_block)
        self.free_blocks.insert(insert_pos, new_free_block)

        # --- Coalescing Logic ---
        # This simple loop merges adjacent blocks after insertion.
        # It might require multiple passes for full coalescing in complex scenarios,
        # but handles basic merging.
        merged = True
        while merged:
            merged = False
            i = 0
            while i < len(self.free_blocks) - 1:
                current_start, current_size = self.free_blocks[i]
                next_start, next_size = self.free_blocks[i + 1]

                if current_start + current_size == next_start:
                    merged_block = (current_start, current_size + next_size)
                    self.free_blocks.pop(i)
                    self.free_blocks.pop(i)  # pop next block (now at index i)
                    self.free_blocks.insert(i, merged_block)
                    merged = True
                    # Restart scan from the beginning of the list after a merge
                    # to ensure all possible merges are caught.
                    # More efficient implementations exist but are more complex.
                    break
                else:
                    i += 1
            # If a merge happened, the outer while loop continues for another pass.

    def __repr__(self) -> str:
        """Provides a string representation of the simulator's state."""
        allocated_mem = sum(alloc["size"] for alloc in self.allocated_blocks.values())
        free_mem = self.available_memory()
        return (
            f"<SimulatedGpuAllocator id='{self._sim_device_id}' name='{self._sim_name}' "
            f"total={self._sim_total_memory} allocated={allocated_mem} free={free_mem} "
            f"fragments={len(self.free_blocks)}>"
        )
