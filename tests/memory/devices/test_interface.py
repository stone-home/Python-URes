# test_mock_device.py
import pytest

# Assuming interface.py is in the same directory or accessible
from ures.memory.devices.interface import DeviceInterface
# Assuming mock_device.py is in the same directory or accessible
from ..conftest import MockDevice


# --- Test Group: Device Properties ---


class TestMockDeviceProperties:
    """Tests focused on the properties of the MockDevice."""

    def test_device_creation_and_properties(self, sample_device: MockDevice):
        """Verify initial properties are set correctly."""
        assert isinstance(sample_device, DeviceInterface) # Check if it implements the interface
        assert sample_device.device_id == "mock-01"
        assert sample_device.name == "TestDevice"
        assert sample_device.total_memory == 1024
        assert sample_device.available_memory == 1024 # Initially, all memory is available

    def test_properties_are_read_only(self, sample_device: MockDevice):
        """Ensure properties are read-only where expected (using @property)."""
        with pytest.raises(AttributeError):
            sample_device.device_id = "new-id" # type: ignore
        with pytest.raises(AttributeError):
            sample_device.name = "NewName" # type: ignore
        with pytest.raises(AttributeError):
            sample_device.total_memory = 2048 # type: ignore
        with pytest.raises(AttributeError):
            sample_device.available_memory = 512 # type: ignore

# --- Test Group: Memory Allocation and Deallocation ---

class TestMockDeviceMemory:
    """Tests focused on the memory operations (malloc, free) of the MockDevice."""

    def test_malloc_successful(self, sample_device: MockDevice):
        """Test allocating a valid block of memory."""
        initial_available = sample_device.available_memory
        alloc_size = 100
        ptr = sample_device.malloc(alloc_size)

        assert ptr is not None # Allocation should succeed
        assert isinstance(ptr, str) # Pointer/handle should be a string
        assert sample_device.available_memory == initial_available - alloc_size
        assert len(sample_device._allocated_blocks) == 1 # Internal check (optional)

    def test_malloc_insufficient_memory(self, sample_device: MockDevice):
        """Test allocating more memory than available."""
        alloc_size = sample_device.total_memory + 1 # More than total
        ptr = sample_device.malloc(alloc_size)

        assert ptr is None # Allocation should fail
        assert sample_device.available_memory == sample_device.total_memory # Memory should not change

    def test_malloc_zero_or_negative_size(self, sample_device: MockDevice):
        """Test allocating zero or negative memory."""
        with pytest.raises(ValueError):
            sample_device.malloc(0)
        with pytest.raises(ValueError):
            sample_device.malloc(-100)
        assert sample_device.available_memory == sample_device.total_memory # Memory should not change

    def test_free_successful(self, sample_device: MockDevice):
        """Test freeing a previously allocated block."""
        alloc_size = 256
        ptr = sample_device.malloc(alloc_size)
        assert ptr is not None
        assert sample_device.available_memory == sample_device.total_memory - alloc_size

        sample_device.free(ptr)

        assert sample_device.available_memory == sample_device.total_memory # Memory should be fully restored
        assert ptr not in sample_device._allocated_blocks # Internal check (optional)

    def test_free_invalid_pointer(self, sample_device: MockDevice):
        """Test freeing a pointer that was never allocated or already freed."""
        initial_available = sample_device.available_memory
        # Try freeing a random non-existent pointer
        sample_device.free("invalid-ptr-123")
        # Memory should remain unchanged if freeing invalid pointers does nothing
        assert sample_device.available_memory == initial_available

        # Allocate and free, then try freeing again
        ptr = sample_device.malloc(50)
        assert ptr is not None
        available_after_alloc = sample_device.available_memory
        sample_device.free(ptr)
        assert sample_device.available_memory == initial_available
        # Try freeing the same pointer again
        sample_device.free(ptr)
        # Memory should still be unchanged
        assert sample_device.available_memory == initial_available

    def test_multiple_allocations_and_frees(self, sample_device: MockDevice):
        """Test a sequence of allocations and frees."""
        ptr1 = sample_device.malloc(100)
        assert ptr1 is not None
        assert sample_device.available_memory == 1024 - 100
        ptr2 = sample_device.malloc(200)
        assert ptr2 is not None
        assert sample_device.available_memory == 1024 - 100 - 200

        sample_device.free(ptr1)
        assert sample_device.available_memory == 1024 - 200 # Only ptr2 is left

        ptr3 = sample_device.malloc(50)
        assert ptr3 is not None
        assert sample_device.available_memory == 1024 - 200 - 50

        sample_device.free(ptr2)
        assert sample_device.available_memory == 1024 - 50 # Only ptr3 is left

        sample_device.free(ptr3)
        assert sample_device.available_memory == 1024 # All freed