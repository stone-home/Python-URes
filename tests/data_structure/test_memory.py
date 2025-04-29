# test_memory_model.py
import pytest
from pydantic import ValidationError
from ures.data_structure.memory import Memory


# Define pytest approx for float comparisons
approx = pytest.approx


class TestMemoryInitialization:
    """Tests for successful Memory model initialization."""

    def test_memory_creation_required_only(self):
        """Test creating a Memory instance with only the required 'bytes' field."""
        mem = Memory(bytes=1024)
        assert mem.bytes == 1024
        assert mem.address is None
        assert mem.alloc_time is None
        assert mem.free_time is None

    def test_memory_creation_all_fields_int_time(self):
        """Test creating a Memory instance with all fields, using integer times."""
        mem = Memory(bytes=2048, address="0x1000", alloc_time=100, free_time=250)
        assert mem.bytes == 2048
        assert mem.address == "0x1000"
        assert mem.alloc_time == 100
        assert mem.free_time == 250

    def test_memory_creation_all_fields_float_time(self):
        """Test creating a Memory instance with all fields, using float times."""
        mem = Memory(bytes=4096, address="0x2000", alloc_time=100.5, free_time=250.75)
        assert mem.bytes == 4096
        assert mem.address == "0x2000"
        assert mem.alloc_time == approx(100.5)
        assert mem.free_time == approx(250.75)


class TestMemoryValidation:
    """Tests for Memory model validation errors."""

    def test_memory_creation_missing_bytes(self):
        """Test that creating a Memory instance without 'bytes' raises ValidationError."""
        with pytest.raises(ValidationError) as excinfo:
            Memory(address="0x3000")
        errors = excinfo.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("bytes",)
        assert errors[0]["type"] == "missing"

    def test_memory_creation_invalid_bytes_type(self):
        """Test that creating a Memory instance with non-integer 'bytes' raises ValidationError."""
        with pytest.raises(ValidationError) as excinfo:
            Memory(bytes="not_an_int")
        errors = excinfo.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("bytes",)
        assert (
            errors[0]["type"] == "int_parsing"
        )  # Or similar type error like 'integer_type'

    def test_memory_creation_invalid_address_type(self):
        """Test that creating a Memory instance with non-string 'address' raises ValidationError."""
        with pytest.raises(ValidationError) as excinfo:
            Memory(bytes=100, address=12345)  # Address should be Optional[str]
        errors = excinfo.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("address",)
        assert errors[0]["type"] == "string_type"

    def test_memory_creation_invalid_time_type(self):
        """Test that creating a Memory instance with non-numeric times raises ValidationError."""
        # Test alloc_time
        with pytest.raises(ValidationError) as excinfo_alloc:
            Memory(bytes=100, alloc_time="not_a_time")

        # Test free_time
        with pytest.raises(ValidationError) as excinfo_free:
            # Using a dict is definitely not int or float
            Memory(bytes=100, free_time={"time": 100})


class TestMemoryProperties:
    """Tests for the calculated properties ('duration', 'is_permanent')."""

    def test_duration_with_int_times(self):
        """Test the duration property with integer alloc_time and free_time."""
        mem = Memory(bytes=100, alloc_time=50, free_time=150)
        assert mem.duration == 100

    def test_duration_with_float_times(self):
        """Test the duration property with float alloc_time and free_time."""
        mem = Memory(bytes=100, alloc_time=50.5, free_time=150.75)
        assert mem.duration == approx(100.25)

    def test_duration_when_not_freed(self):
        """Test the duration property when free_time is None."""
        mem = Memory(bytes=100, alloc_time=50, free_time=None)
        assert mem.duration is None

    def test_duration_when_alloc_time_is_none(self):
        """Test the duration property calculation raises TypeError if alloc_time is None but free_time is set."""
        mem = Memory(bytes=100, alloc_time=None, free_time=150)
        # Accessing duration should raise TypeError because free_time - None is invalid
        with pytest.raises(TypeError) as excinfo:
            _ = mem.duration  # Access the property
        assert "unsupported operand type(s) for -" in str(excinfo.value)

    def test_duration_when_both_times_none(self):
        """Test the duration property when both alloc_time and free_time are None."""
        mem = Memory(bytes=100, alloc_time=None, free_time=None)
        assert mem.duration is None  # free_time is None, so duration is None

    def test_is_permanent_when_freed(self):
        """Test is_permanent property when free_time is set."""
        mem = Memory(bytes=100, alloc_time=50, free_time=150)
        assert mem.is_permanent is False

    def test_is_permanent_when_not_freed(self):
        """Test is_permanent property when free_time is None."""
        mem = Memory(bytes=100, alloc_time=50, free_time=None)
        assert mem.is_permanent is True


class TestMemoryRepr:
    """Tests for the __repr__ method."""

    def test_repr_minimal(self):
        """Test the __repr__ output with minimal data."""
        mem = Memory(bytes=512)
        expected = "Memory(None)|512 bytes|None->None|dur: None"
        assert repr(mem) == expected

    def test_repr_all_data(self):
        """Test the __repr__ output with all data."""
        mem = Memory(bytes=1024, address="0xA000", alloc_time=10, free_time=25.5)
        # Duration is 15.5
        expected = "Memory(0xA000)|1024 bytes|10->25.5|dur: 15.5"
        assert repr(mem) == expected

    def test_repr_not_freed(self):
        """Test the __repr__ output when the block is not freed."""
        mem = Memory(bytes=2048, address="0xB000", alloc_time=50)
        # Duration is None
        expected = "Memory(0xB000)|2048 bytes|50->None|dur: None"
        assert repr(mem) == expected
