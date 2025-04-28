import uuid
import pytest
import time
from typing import Optional, Dict
# Assuming interface.py is in the same directory or accessible in the Python path
from ures.memory.devices.interface import DeviceInterface
from ures.memory.type import MemoryBlock
from ures.memory.manager import MemoryManager


class MockDevice(DeviceInterface):
    """
    A concrete mock implementation of the DeviceInterface for testing purposes.
    """
    def __init__(self, device_id: str, name: str, total_memory: int):
        self._device_id = device_id
        self._name = name
        self._total_memory = total_memory
        self._available_memory = total_memory
        self._allocated_blocks: Dict[str, int] = {} # Store ptr -> size

    @property
    def device_id(self) -> str:
        return self._device_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def total_memory(self) -> int:
        return self._total_memory

    @property
    def available_memory(self) -> int:
        return self._available_memory

    def malloc(self, size: int) -> Optional[str]:
        """
        Simple mock allocation: allocates if enough memory is available.
        Returns a unique identifier as the 'pointer'.
        """
        if size <= 0:
            raise ValueError("Allocation size must be positive")
        if size > self.available_memory:
            return None # Allocation failed

        self._available_memory -= size
        # Generate a simple unique handle (UUID) as the memory pointer/handle
        ptr = str(uuid.uuid4())
        self._allocated_blocks[ptr] = size
        return ptr

    def free(self, ptr: str):
        """
        Simple mock deallocation: frees the block if the pointer is valid.
        """
        if ptr in self._allocated_blocks:
            size = self._allocated_blocks.pop(ptr)
            self._available_memory += size
        else:
            # Optional: Raise an error for trying to free an invalid pointer,
            # depending on desired device behavior.
            # raise ValueError(f"Invalid pointer: {ptr}")
            # Or simply do nothing if that's the expected behavior
            pass


@pytest.fixture
def sample_device() -> MockDevice:
    """Provides a standard MockDevice instance for tests."""
    return MockDevice(device_id="mock-01", name="TestDevice", total_memory=1024)

# --- Test Fixtures created for type ---

@pytest.fixture
def mock_device_gpu() -> MockDevice:
    """Provides a standard MockDevice instance."""
    return MockDevice(device_id="mock-gpu-01", name="TestGPU", total_memory=4096)

@pytest.fixture
def mock_device_cpu() -> MockDevice:
    """Provides an alternative MockDevice instance."""
    return MockDevice(device_id="mock-cpu-02", name="TestCPU", total_memory=8192)

@pytest.fixture(autouse=True)
def patch_time(monkeypatch):
    """Patch time.time_ns for predictable timestamps in tests."""
    # Using autouse=True to apply it to all tests in this module
    monkeypatch.setattr(time, 'time_ns', lambda: 1234567890000000000) # Example fixed time

@pytest.fixture
def memory_block_basic(mock_device_gpu: MockDevice) -> MemoryBlock:
    """Provides a MemoryBlock with minimal initialization."""
    return MemoryBlock(byte=1024, device=mock_device_gpu)

@pytest.fixture
def memory_block_full(mock_device_gpu: MockDevice) -> MemoryBlock:
    """Provides a MemoryBlock with all optional arguments."""
    return MemoryBlock(
        byte=2048,
        device=mock_device_gpu,
        actual_bytes=2000,
        address="0x1234abcd",
        alloc_time=1678886400.0, # Example timestamp (float)
        free_time=1678887400,   # Example timestamp (int)
        comment="Test block"
    )


# --- Test Fixtures created for sequence ---
@pytest.fixture
def mock_device() -> MockDevice:
    """Provides a standard MockDevice instance for tests."""
    return MockDevice(device_id="mock_id", name="mock_device", total_memory=1024)

@pytest.fixture
def mb_alloc_only(mock_device: MockDevice) -> MemoryBlock:
    """Memory block with only allocation time."""
    return MemoryBlock(byte=100, device=mock_device, alloc_time=10)

@pytest.fixture
def mb_alloc_free(mock_device: MockDevice) -> MemoryBlock:
    """Memory block with allocation and free times."""
    return MemoryBlock(byte=200, device=mock_device, alloc_time=5, free_time=15)

@pytest.fixture
def mb_alloc_none(mock_device: MockDevice) -> MemoryBlock:
    """Memory block with alloc_time=None (should become time 0)."""
    return MemoryBlock(byte=50, device=mock_device, alloc_time=None, free_time=8)

@pytest.fixture
def mb_free_none(mock_device: MockDevice) -> MemoryBlock:
    """Memory block with free_time=None (should only generate alloc)."""
    return MemoryBlock(byte=75, device=mock_device, alloc_time=3)

@pytest.fixture
def mb_alloc_free_same_time(mock_device: MockDevice) -> MemoryBlock:
    """Memory block with alloc and free at the same time."""
    return MemoryBlock(byte=150, device=mock_device, alloc_time=12, free_time=12)


# --- Test Fixtures created for manager ---

@pytest.fixture
def memory_block_A(mock_device: 'MockDevice') -> MemoryBlock: # Use type hint from conftest if defined there
    """A sample memory block using the standard mock device."""
    # Pass necessary kwargs if the real MemoryBlock requires them beyond byte/device
    return MemoryBlock(byte=100, device=mock_device, alloc_time=10, free_time=20, address="addr_A")

@pytest.fixture
def memory_block_B(mock_device: 'MockDevice') -> MemoryBlock:
    """Another sample memory block using the standard mock device."""
    return MemoryBlock(byte=200, device=mock_device, alloc_time=15, address="addr_B") # No free time

@pytest.fixture
def memory_block_C(mock_device: 'MockDevice') -> MemoryBlock:
    """A third sample memory block using the standard mock device."""
    return MemoryBlock(byte=50, device=mock_device, alloc_time=5, free_time=25, address="addr_C")


@pytest.fixture
def empty_manager() -> MemoryManager:
    """An empty MemoryManager instance."""
    return MemoryManager()

@pytest.fixture
def populated_manager(empty_manager: MemoryManager, memory_block_A: MemoryBlock, memory_block_B: MemoryBlock) -> MemoryManager:
    """A MemoryManager instance pre-populated with blocks A and B."""
    empty_manager.add(memory_block_A)
    empty_manager.add(memory_block_B)
    return empty_manager