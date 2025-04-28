import pytest
from ures.data_structure.memory import AbsMemoryBlock
from ures.memory.type import MemoryBlock
from .conftest import MockDevice



# --- Test Group: Initialization ---

class TestMemoryBlockInitialization:
    """Tests focused on the __init__ method of MemoryBlock."""

    def test_basic_initialization(self, memory_block_basic: MemoryBlock, mock_device_gpu: MockDevice):
        """Verify basic properties are set correctly using real AbsMemoryBlock."""
        assert isinstance(memory_block_basic, MemoryBlock)
        # Check inheritance using the *real* AbsMemoryBlock
        assert isinstance(memory_block_basic, AbsMemoryBlock)
        assert memory_block_basic.bytes == 1024
        assert memory_block_basic.device == mock_device_gpu
        # Check defaults for optional args
        assert memory_block_basic.allocated_size is None
        assert memory_block_basic.address is None
        assert memory_block_basic.alloc_time is None
        assert memory_block_basic.free_time is None
        assert memory_block_basic.comment == ""

    def test_full_initialization(self, memory_block_full: MemoryBlock, mock_device_gpu: MockDevice):
        """Verify properties when all optional args are provided."""
        assert memory_block_full.bytes == 2048
        assert memory_block_full.device == mock_device_gpu
        assert memory_block_full.allocated_size == 2000
        assert memory_block_full.address == "0x1234abcd"
        assert memory_block_full.alloc_time == 1678886400.0
        assert memory_block_full.free_time == 1678887400
        assert memory_block_full.comment == "Test block"

# --- Test Group: Properties ---

class TestMemoryBlockProperties:
    """Tests focused on the getters and setters of MemoryBlock."""

    def test_read_only_properties(self, memory_block_basic: MemoryBlock):
        """Verify that properties without setters are read-only."""
        with pytest.raises(AttributeError):
            memory_block_basic.bytes = 512 # type: ignore
        with pytest.raises(AttributeError):
            memory_block_basic.allocated_size = 500 # type: ignore

    def test_address_setter(self, memory_block_basic: MemoryBlock):
        """Test setting the address property (relies on real type_check)."""
        memory_block_basic.address = "0xABCDEF00"
        assert memory_block_basic.address == "0xABCDEF00"
        memory_block_basic.address = None
        assert memory_block_basic.address is None
        # This test now depends on the *actual* behavior of type_check
        # Assuming it raises TypeError for incorrect types:
        with pytest.raises(TypeError):
             memory_block_basic.address = 12345 # type: ignore

    def test_alloc_time_setter(self, memory_block_basic: MemoryBlock):
        """Test setting the alloc_time property."""
        fixed_time = 1700000000.5
        memory_block_basic.alloc_time = fixed_time
        assert memory_block_basic.alloc_time == fixed_time

        # Test setting None uses current time (mocked via patch_time fixture)
        memory_block_basic.alloc_time = None
        assert memory_block_basic.alloc_time == 1234567890000000000 # Mocked time_ns

    def test_free_time_setter(self, memory_block_basic: MemoryBlock):
        """Test setting the free_time property (relies on real type_check)."""
        fixed_time = 1710000000
        memory_block_basic.free_time = fixed_time
        assert memory_block_basic.free_time == fixed_time

        # Test setting None uses current time (mocked via patch_time fixture)
        memory_block_basic.free_time = None
        assert memory_block_basic.free_time == 1234567890000000000 # Mocked time_ns

        # This test now depends on the *actual* behavior of type_check
        # Assuming it raises TypeError for incorrect types:
        with pytest.raises(TypeError):
            memory_block_basic.free_time = "not a time" # type: ignore

    def test_comment_setter(self, memory_block_basic: MemoryBlock):
        """Test setting the comment property (relies on real type_check)."""
        memory_block_basic.comment = "New comment"
        assert memory_block_basic.comment == "New comment"
        # This test now depends on the *actual* behavior of type_check
        # Assuming it raises TypeError for incorrect types:
        with pytest.raises(TypeError):
             memory_block_basic.comment = 123 # type: ignore

    def test_device_setter(self, memory_block_basic: MemoryBlock, mock_device_cpu: MockDevice):
        """Test setting the device property (relies on real type_check)."""
        memory_block_basic.device = mock_device_cpu
        assert memory_block_basic.device == mock_device_cpu
        assert memory_block_basic.device.name == "TestCPU"
        # This test now depends on the *actual* behavior of type_check
        # Assuming it raises TypeError for incorrect types:
        with pytest.raises(TypeError):
             memory_block_basic.device = "not a device" # type: ignore

# --- Test Group: Representation ---

class TestMemoryBlockRepresentation:
    """Tests focused on the __str__ and __repr__ methods."""

    def test_repr_method(self, memory_block_full: MemoryBlock):
        """Test the __repr__ output format."""
        # Assumes mock_device_gpu has name "TestGPU"
        expected_repr = "TestGPU|0x1234abcd|Test block|2048|1678886400.0|1678887400"
        assert repr(memory_block_full) == expected_repr

    def test_repr_method_defaults(self, memory_block_basic: MemoryBlock):
        """Test the __repr__ output format with default values."""
        # Set alloc/free time using the setter which now uses patched time_ns
        memory_block_basic.alloc_time = None
        memory_block_basic.free_time = None
        # Assumes mock_device_gpu has name "TestGPU"
        expected_repr = "TestGPU|None||1024|1234567890000000000|1234567890000000000"
        assert repr(memory_block_basic) == expected_repr

    def test_str_method(self, memory_block_full: MemoryBlock):
        """Test the __str__ output format (relies on real format_memory)."""
        # The output depends entirely on the *actual* format_memory function
        # Example: If format_memory(2048) returns "2.0 KiB"
        # expected_str = "2.0 KiB in Address 0x1234abcd (Test block)"
        # You'll need to know the expected output of your real format_memory
        # For now, just check it produces a string without error
        assert isinstance(str(memory_block_full), str)
        assert "0x1234abcd" in str(memory_block_full)
        assert "(Test block)" in str(memory_block_full)
        assert "Address" in str(memory_block_full)
        # Add more specific assertion based on format_memory's known behavior

    def test_str_method_defaults(self, memory_block_basic: MemoryBlock):
        """Test the __str__ output format with default values (relies on real format_memory)."""
        # Example: If format_memory(1024) returns "1.0 KiB"
        # expected_str = "1.0 KiB in Address None ()"
        # Check basic structure
        assert isinstance(str(memory_block_basic), str)
        assert "Address None" in str(memory_block_basic)
        assert "()" in str(memory_block_basic)
        # Add more specific assertion based on format_memory's known behavior