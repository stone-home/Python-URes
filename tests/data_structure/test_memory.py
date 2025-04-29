# test_memory_model.py
import pytest
from typing import Union, Optional
from pydantic import ValidationError
from ures.data_structure.memory import Memory, MemoryBlockInterface


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
        memory = "0x1000"
        mem = Memory(bytes=2048, address=int(memory, 16), alloc_time=100, free_time=250)
        assert mem.bytes == 2048
        assert mem.address == int(memory, 16)
        assert mem.alloc_time == 100
        assert mem.free_time == 250

    def test_memory_creation_all_fields_float_time(self):
        """Test creating a Memory instance with all fields, using float times."""
        memory = "0x2000"
        mem = Memory(
            bytes=4096, address=int(memory, 16), alloc_time=100.5, free_time=250.75
        )
        assert mem.bytes == 4096
        assert mem.address == int(memory, 16)
        assert mem.alloc_time == approx(100.5)
        assert mem.free_time == approx(250.75)


class TestMemoryValidation:
    """Tests for Memory model validation errors."""

    def test_memory_creation_missing_bytes(self):
        """Test that creating a Memory instance without 'bytes' raises ValidationError."""
        with pytest.raises(ValidationError) as excinfo:
            Memory(address=int("0x3000", 16))
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
            Memory(bytes=100, address="0x8721")  # Address should be Optional[str]
        errors = excinfo.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("address",)
        assert errors[0]["type"] == "int_parsing"

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
        memory = "0xa000"
        mem = Memory(bytes=1024, address=int(memory, 16), alloc_time=10, free_time=25.5)
        # Duration is 15.5
        expected = f"Memory({memory})|1024 bytes|10->25.5|dur: 15.5"
        assert repr(mem) == expected

    def test_repr_not_freed(self):
        """Test the __repr__ output when the block is not freed."""
        memory = "0xb000"
        mem = Memory(bytes=2048, address=int(memory, 16), alloc_time=50)
        # Duration is None
        expected = f"Memory({memory})|2048 bytes|50->None|dur: None"
        assert repr(mem) == expected


# ------------------------- TestCases for MemoryBlockInterface ------------------------- #
class ConcreteMemoryBlockV2(MemoryBlockInterface):
    """A concrete implementation of the fully abstract interface."""

    def __init__(
        self,
        b: int,
        addr: str,
        alloc_t: Union[int, float],
        free_t: Optional[Union[int, float]] = None,
    ):
        self._b = b
        self._addr = addr
        self._alloc_t = alloc_t
        self._free_t = free_t

    @property
    def bytes(self) -> int:
        return self._b

    @property
    def address(self) -> str:
        return self._addr

    @property
    def alloc_time(self) -> Union[int, float]:
        return self._alloc_t

    @property
    def free_time(self) -> Optional[Union[int, float]]:
        return self._free_t

    # --- Must implement duration and is_permanent now ---
    @property
    def duration(self) -> Optional[Union[int, float]]:
        # Example implementation consistent with previous logic
        if self._alloc_t is None or self._free_t is None:
            return None
        if isinstance(self._alloc_t, (int, float)) and isinstance(
            self._free_t, (int, float)
        ):
            return self._free_t - self._alloc_t
        return None  # Should not happen with valid inputs

    @property
    def is_permanent(self) -> bool:
        # Example implementation consistent with previous logic
        return self._free_t is None


class TestInterfaceAbstractness:
    """Tests the abstract nature of the MemoryBlockInterface."""

    def test_cannot_instantiate_abc_directly(self):
        """Verify that the ABC itself cannot be instantiated."""
        with pytest.raises(TypeError) as excinfo:
            MemoryBlockInterface()  # Attempt to instantiate the ABC
        assert "Can't instantiate abstract class" in str(excinfo.value)
        # Check for at least one of the newly abstract methods
        assert "duration" in str(excinfo.value) or "is_permanent" in str(excinfo.value)

    def test_can_instantiate_concrete_class(self):
        """Verify a correctly implemented concrete class can be instantiated."""
        try:
            block = ConcreteMemoryBlockV2(b=100, addr="0x1", alloc_t=10, free_t=20)
            assert isinstance(block, MemoryBlockInterface)
            # Check if properties are accessible
            assert block.bytes == 100
            assert block.duration == 10
            assert block.is_permanent is False
        except Exception as e:
            pytest.fail(f"Failed to instantiate or use ConcreteMemoryBlockV2: {e}")

    def test_concrete_class_missing_method_fails(self):
        """Verify a concrete class missing an abstract method raises TypeError on instantiation."""

        class IncompleteBlock(MemoryBlockInterface):
            # Missing 'bytes' implementation
            @property
            def address(self):
                return "0xINC"

            @property
            def alloc_time(self):
                return 10

            @property
            def free_time(self):
                return None

            @property
            def duration(self):
                return None

            @property
            def is_permanent(self):
                return True

        with pytest.raises(TypeError) as excinfo:
            IncompleteBlock()
        assert "Can't instantiate abstract class" in str(excinfo.value)
        assert "bytes" in str(excinfo.value)  # Specifically missing 'bytes'
