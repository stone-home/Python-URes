"""
Shared pytest configuration and fixtures for memory allocator tests.
"""

import pytest
from typing import Dict, Any

from ures.memory.allocator import DeviceMemorySimulator, FirstFitAllocator
from ures.memory.blocks import BlockPool


@pytest.fixture(scope="session")
def test_config():
    """Session-wide test configuration"""
    return {
        "default_device_size": 8192,
        "default_base_address": 0x10000,
        "test_allocation_sizes": [64, 128, 256, 512, 1024, 2048, 4096],
        "small_device_size": 1024,
        "large_device_size": 65536,
    }


@pytest.fixture
def small_device():
    """Create a small device for basic testing"""
    return DeviceMemorySimulator(device_id=0, total_memory=1024, base_address=0x1000)


@pytest.fixture
def standard_device():
    """Create a standard-sized device for most tests"""
    return DeviceMemorySimulator(device_id=0, total_memory=8192, base_address=0x10000)


@pytest.fixture
def large_device():
    """Create a large device for performance testing"""
    return DeviceMemorySimulator(device_id=0, total_memory=65536, base_address=0x100000)


@pytest.fixture
def empty_pool():
    """Create an empty BlockPool for testing"""
    return BlockPool()


@pytest.fixture
def pool_with_segment():
    """Create a BlockPool with a single segment"""
    pool = BlockPool()
    segment = pool.create_segment(0x1000, 4096, device=0)
    return pool


@pytest.fixture
def fragmented_pool():
    """Create a BlockPool with fragmented memory for testing allocators"""
    pool = BlockPool()
    segment = pool.create_segment(0x1000, 8192, device=0)

    # Create some allocations and free some to create fragmentation
    allocator = FirstFitAllocator()

    # Allocate blocks
    allocs = []
    sizes = [512, 256, 1024, 128, 512, 256, 1024]
    for size in sizes:
        from ures.memory.allocator import AllocationRequest, FreeRequest

        result = allocator.allocate(pool, AllocationRequest(size=size))
        if result.success:
            allocs.append((result.address, size))

    # Free every other allocation to create gaps
    for i, (addr, size) in enumerate(allocs):
        if i % 2 == 0:
            allocator.free(pool, FreeRequest(address=addr))

    return pool


def pytest_configure(config):
    """Configure pytest with custom markers and settings"""
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add automatic markers"""
    for item in items:
        # Add unit marker to tests by default
        if not any(item.iter_markers()):
            item.add_marker(pytest.mark.unit)

        # Add slow marker to performance tests
        if item.get_closest_marker("performance"):
            item.add_marker(pytest.mark.slow)


@pytest.fixture(autouse=True)
def reset_allocator_state():
    """Automatically reset allocator state between tests"""
    yield


# Any cleanup code can go here if needed


# Helper functions for tests
def create_allocation_pattern(sizes, streams=None):
    """Helper to create allocation patterns for testing"""
    if streams is None:
        streams = [None] * len(sizes)
    return list(zip(sizes, streams))


def verify_allocation_result(result, expected_size=None, should_succeed=True):
    """Helper to verify allocation results"""
    assert result is not None
    assert hasattr(result, "success")

    if should_succeed:
        assert result.success is True
        assert result.address is not None
        if expected_size:
            assert result.actual_size == expected_size
    else:
        assert result.success is False


def verify_free_result(result, expected_size=None, should_succeed=True):
    """Helper to verify free results"""
    assert result is not None
    assert hasattr(result, "success")

    if should_succeed:
        assert result.success is True
        if expected_size:
            assert result.freed_size == expected_size
    else:
        assert result.success is False


# Performance testing utilities
class PerformanceTimer:
    """Context manager for timing operations"""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.duration = None

    def __enter__(self):
        self.start_time = pytest.time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = pytest.time.time()
        self.duration = self.end_time - self.start_time


@pytest.fixture
def performance_timer():
    """Provide a performance timer for tests"""
    return PerformanceTimer


# Memory usage testing utilities
def get_memory_usage_stats(device):
    """Get memory usage statistics for a device"""
    info = device.get_memory_info()
    summary = info["memory_summary"]

    return {
        "total_memory": device.total_memory,
        "allocated_bytes": summary["total_allocated_bytes"],
        "free_bytes": summary["total_free_bytes"],
        "utilization": summary["overall_utilization"],
        "fragmentation": summary["average_fragmentation"],
        "allocation_count": info["allocation_count"],
        "free_count": info["free_count"],
    }


# Test data generators
@pytest.fixture
def allocation_sizes():
    """Common allocation sizes for testing"""
    return [64, 128, 256, 512, 1024, 2048, 4096]


@pytest.fixture
def random_allocation_pattern():
    """Generate a random allocation pattern for stress testing"""
    import random

    random.seed(42)  # Deterministic for reproducible tests

    sizes = []
    for _ in range(50):
        size = random.choice([64, 128, 256, 512, 1024])
        sizes.append(size)

    return sizes


# Custom assertions
def assert_memory_consistency(device):
    """Assert that device memory state is consistent"""
    info = device.get_memory_info()
    summary = info["memory_summary"]

    # Basic consistency checks
    assert summary["total_allocated_bytes"] >= 0
    assert summary["total_free_bytes"] >= 0
    assert (
        summary["total_allocated_bytes"] + summary["total_free_bytes"]
        <= device.total_memory
    )
    assert 0.0 <= summary["overall_utilization"] <= 1.0
    assert 0.0 <= summary["average_fragmentation"] <= 1.0


def assert_allocator_stats_consistency(allocator):
    """Assert that allocator statistics are consistent"""
    stats = allocator.get_statistics()

    assert stats["allocation_count"] >= 0
    assert stats["free_count"] >= 0
    assert stats["total_allocated"] >= 0
    assert stats["total_freed"] >= 0
    assert (
        stats["currently_allocated"] == stats["total_allocated"] - stats["total_freed"]
    )
    assert stats["active_blocks"] >= 0
    assert len(stats["allocation_times"]) == stats["allocation_count"]
    assert len(stats["free_times"]) == stats["free_count"]
