#!/usr/bin/env python3
"""
Comprehensive test suite for all memory allocation algorithms.

This module provides detailed testing and comparison of all implemented
allocation algorithms including performance metrics, fragmentation analysis,
and behavioral verification.
"""

import time
import random
import statistics
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Import the memory allocation system
from ures.memory.allocator import (
    DeviceMemorySimulator,
    FirstFitAllocator,
    BestFitAllocator,
    WorstFitAllocator,
    NextFitAllocator,
    BuddySystemAllocator,
    AllocationRequest,
    FreeRequest,
)


class TestType(Enum):
    """Types of tests to run"""

    BASIC_FUNCTIONALITY = "basic_functionality"
    PERFORMANCE = "performance"
    FRAGMENTATION = "fragmentation"
    STRESS = "stress"
    EDGE_CASES = "edge_cases"
    COMPARISON = "comparison"


@dataclass
class TestResult:
    """Results from running a test"""

    test_name: str
    algorithm: str
    success: bool
    execution_time_ms: float
    memory_utilization: float
    fragmentation_ratio: float
    allocations_successful: int
    allocations_failed: int
    details: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class AllocationPattern:
    """Defines an allocation pattern for testing"""

    name: str
    sizes: List[int]
    streams: List[Optional[int]]
    free_indices: List[int]  # Which allocations to free
    description: str


class AlgorithmTester:
    """Main class for testing memory allocation algorithms"""

    def __init__(self, device_size: int = 64 * 1024, base_address: int = 0x100000):
        self.device_size = device_size
        self.base_address = base_address
        self.test_results: List[TestResult] = []
        self.algorithms = [
            "First Fit",
            "Best Fit",
            "Worst Fit",
            "Next Fit",
            "Buddy System",
        ]

        # Predefined test patterns
        self.test_patterns = self._create_test_patterns()

    def _create_test_patterns(self) -> List[AllocationPattern]:
        """Create various allocation patterns for testing"""
        return [
            AllocationPattern(
                name="Sequential Small",
                sizes=[64] * 20,
                streams=[None] * 20,
                free_indices=[1, 3, 5, 7, 9],
                description="Small sequential allocations with some frees",
            ),
            AllocationPattern(
                name="Sequential Large",
                sizes=[1024] * 10,
                streams=[None] * 10,
                free_indices=[2, 4, 6],
                description="Large sequential allocations",
            ),
            AllocationPattern(
                name="Mixed Sizes",
                sizes=[64, 128, 256, 512, 1024, 2048, 128, 256, 64, 512],
                streams=[None] * 10,
                free_indices=[1, 3, 7],
                description="Mixed allocation sizes",
            ),
            AllocationPattern(
                name="Power of 2",
                sizes=[64, 128, 256, 512, 1024, 2048, 4096],
                streams=[None] * 7,
                free_indices=[1, 3, 5],
                description="Power-of-2 sizes (optimal for buddy system)",
            ),
            AllocationPattern(
                name="Non-Power of 2",
                sizes=[100, 300, 700, 900, 1500, 2100],
                streams=[None] * 6,
                free_indices=[1, 3],
                description="Non-power-of-2 sizes (challenging for buddy system)",
            ),
            AllocationPattern(
                name="Multi-Stream",
                sizes=[256, 512, 256, 512, 1024, 256],
                streams=[1, 2, 1, 2, 3, 1],
                free_indices=[1, 4],
                description="Multiple streams allocation",
            ),
            AllocationPattern(
                name="Alternating Free",
                sizes=[256] * 16,
                streams=[None] * 16,
                free_indices=list(range(0, 16, 2)),  # Free every other allocation
                description="Alternating allocation and free pattern",
            ),
            AllocationPattern(
                name="Fragmentation Test",
                sizes=[512, 256, 512, 256, 512, 256, 512],
                streams=[None] * 7,
                free_indices=[1, 3, 5],  # Free all 256-byte blocks
                description="Creates specific fragmentation pattern",
            ),
        ]

    def create_device(self) -> DeviceMemorySimulator:
        """Create a fresh device for testing"""
        return DeviceMemorySimulator(
            device_id=0, total_memory=self.device_size, base_address=self.base_address
        )

    def run_basic_functionality_test(
        self, algorithm: str, pattern: AllocationPattern
    ) -> TestResult:
        """Test basic allocation and deallocation functionality"""
        device = self.create_device()
        device.set_allocator(algorithm)

        start_time = time.time()

        try:
            allocated_addresses = []
            successful_allocations = 0
            failed_allocations = 0

            # Perform allocations
            for i, (size, stream) in enumerate(zip(pattern.sizes, pattern.streams)):
                result = device.allocate(size=size, stream=stream)
                if result.success:
                    allocated_addresses.append(result.address)
                    successful_allocations += 1
                else:
                    failed_allocations += 1

            # Perform frees
            for idx in pattern.free_indices:
                if idx < len(allocated_addresses):
                    free_result = device.free(allocated_addresses[idx])
                    if not free_result.success:
                        print(
                            f"Warning: Failed to free address {hex(allocated_addresses[idx])}"
                        )

            execution_time = (time.time() - start_time) * 1000
            memory_info = device.get_memory_info()

            return TestResult(
                test_name=f"Basic Functionality - {pattern.name}",
                algorithm=algorithm,
                success=True,
                execution_time_ms=execution_time,
                memory_utilization=memory_info["memory_summary"]["overall_utilization"],
                fragmentation_ratio=memory_info["memory_summary"][
                    "average_fragmentation"
                ],
                allocations_successful=successful_allocations,
                allocations_failed=failed_allocations,
                details={
                    "pattern": pattern.name,
                    "total_allocations": len(pattern.sizes),
                    "total_frees": len(pattern.free_indices),
                    "allocated_addresses": len(allocated_addresses),
                    "allocator_stats": device.get_allocator_statistics()[algorithm],
                },
            )

        except Exception as e:
            return TestResult(
                test_name=f"Basic Functionality - {pattern.name}",
                algorithm=algorithm,
                success=False,
                execution_time_ms=(time.time() - start_time) * 1000,
                memory_utilization=0.0,
                fragmentation_ratio=0.0,
                allocations_successful=0,
                allocations_failed=0,
                details={},
                error_message=str(e),
            )

    def run_performance_test(
        self, algorithm: str, num_operations: int = 1000
    ) -> TestResult:
        """Test allocation/deallocation performance under load"""
        device = self.create_device()
        device.set_allocator(algorithm)

        random.seed(42)  # Deterministic for reproducible results

        start_time = time.time()

        try:
            allocated_blocks = []
            successful_allocations = 0
            failed_allocations = 0
            allocation_times = []
            free_times = []

            for i in range(num_operations):
                # 70% chance to allocate, 30% chance to free (if blocks available)
                if random.random() < 0.7 or not allocated_blocks:
                    # Allocate
                    size = random.choice([64, 128, 256, 512, 1024])

                    alloc_start = time.time_ns()
                    result = device.allocate(size=size)
                    alloc_end = time.time_ns()

                    allocation_times.append(
                        (alloc_end - alloc_start) / 1000
                    )  # microseconds

                    if result.success:
                        allocated_blocks.append(result.address)
                        successful_allocations += 1
                    else:
                        failed_allocations += 1
                else:
                    # Free
                    address = random.choice(allocated_blocks)
                    allocated_blocks.remove(address)

                    free_start = time.time_ns()
                    free_result = device.free(address)
                    free_end = time.time_ns()

                    free_times.append((free_end - free_start) / 1000)  # microseconds

            execution_time = (time.time() - start_time) * 1000
            memory_info = device.get_memory_info()

            return TestResult(
                test_name="Performance Test",
                algorithm=algorithm,
                success=True,
                execution_time_ms=execution_time,
                memory_utilization=memory_info["memory_summary"]["overall_utilization"],
                fragmentation_ratio=memory_info["memory_summary"][
                    "average_fragmentation"
                ],
                allocations_successful=successful_allocations,
                allocations_failed=failed_allocations,
                details={
                    "num_operations": num_operations,
                    "avg_allocation_time_us": (
                        statistics.mean(allocation_times) if allocation_times else 0
                    ),
                    "max_allocation_time_us": (
                        max(allocation_times) if allocation_times else 0
                    ),
                    "avg_free_time_us": (
                        statistics.mean(free_times) if free_times else 0
                    ),
                    "max_free_time_us": max(free_times) if free_times else 0,
                    "operations_per_second": num_operations / (execution_time / 1000),
                    "active_blocks_remaining": len(allocated_blocks),
                },
            )

        except Exception as e:
            return TestResult(
                test_name="Performance Test",
                algorithm=algorithm,
                success=False,
                execution_time_ms=(time.time() - start_time) * 1000,
                memory_utilization=0.0,
                fragmentation_ratio=0.0,
                allocations_successful=0,
                allocations_failed=0,
                details={},
                error_message=str(e),
            )

    def run_fragmentation_test(self, algorithm: str) -> TestResult:
        """Test how well algorithm handles fragmentation"""
        device = self.create_device()
        device.set_allocator(algorithm)

        start_time = time.time()

        try:
            # Phase 1: Fill memory with alternating large and small blocks
            allocated_large = []
            allocated_small = []

            for i in range(20):
                # Allocate large block
                result_large = device.allocate(size=2048)
                if result_large.success:
                    allocated_large.append(result_large.address)

                # Allocate small block
                result_small = device.allocate(size=128)
                if result_small.success:
                    allocated_small.append(result_small.address)

            initial_utilization = device.get_memory_info()["memory_summary"][
                "overall_utilization"
            ]

            # Phase 2: Free all large blocks to create fragmentation
            for addr in allocated_large:
                device.free(addr)

            fragmented_utilization = device.get_memory_info()["memory_summary"][
                "overall_utilization"
            ]
            fragmentation_after_free = device.get_memory_info()["memory_summary"][
                "average_fragmentation"
            ]

            # Phase 3: Try to allocate medium-sized blocks that should fit in gaps
            medium_successful = 0
            medium_failed = 0

            for i in range(10):
                result = device.allocate(size=1024)  # Should fit in 2048-byte gaps
                if result.success:
                    medium_successful += 1
                else:
                    medium_failed += 1

            execution_time = (time.time() - start_time) * 1000
            final_memory_info = device.get_memory_info()

            return TestResult(
                test_name="Fragmentation Test",
                algorithm=algorithm,
                success=True,
                execution_time_ms=execution_time,
                memory_utilization=final_memory_info["memory_summary"][
                    "overall_utilization"
                ],
                fragmentation_ratio=final_memory_info["memory_summary"][
                    "average_fragmentation"
                ],
                allocations_successful=len(allocated_large)
                + len(allocated_small)
                + medium_successful,
                allocations_failed=medium_failed,
                details={
                    "initial_utilization": initial_utilization,
                    "fragmented_utilization": fragmented_utilization,
                    "fragmentation_after_free": fragmentation_after_free,
                    "large_blocks_allocated": len(allocated_large),
                    "small_blocks_allocated": len(allocated_small),
                    "medium_blocks_successful": medium_successful,
                    "medium_blocks_failed": medium_failed,
                    "fragmentation_resistance": medium_successful
                    / 10.0,  # Ratio of successful medium allocations
                },
            )

        except Exception as e:
            return TestResult(
                test_name="Fragmentation Test",
                algorithm=algorithm,
                success=False,
                execution_time_ms=(time.time() - start_time) * 1000,
                memory_utilization=0.0,
                fragmentation_ratio=0.0,
                allocations_successful=0,
                allocations_failed=0,
                details={},
                error_message=str(e),
            )

    def run_stress_test(self, algorithm: str, duration_seconds: int = 5) -> TestResult:
        """Run algorithm under stress for specified duration"""
        device = self.create_device()
        device.set_allocator(algorithm)

        start_time = time.time()
        end_time = start_time + duration_seconds

        try:
            allocated_blocks = []
            successful_allocations = 0
            failed_allocations = 0
            operations_performed = 0

            while time.time() < end_time:
                operations_performed += 1

                # Random operation: allocate or free
                if (
                    random.random() < 0.6 or not allocated_blocks
                ):  # 60% allocate, 40% free
                    # Allocate random size
                    size = random.randint(32, 4096)
                    result = device.allocate(size=size)

                    if result.success:
                        allocated_blocks.append(result.address)
                        successful_allocations += 1
                    else:
                        failed_allocations += 1

                        # If allocation failed due to fragmentation, try smaller size
                        if "No suitable block found" in (result.error_message or ""):
                            small_result = device.allocate(size=64)
                            if small_result.success:
                                allocated_blocks.append(small_result.address)
                                successful_allocations += 1
                else:
                    # Free random block
                    if allocated_blocks:
                        address = random.choice(allocated_blocks)
                        allocated_blocks.remove(address)
                        device.free(address)

            execution_time = (time.time() - start_time) * 1000
            memory_info = device.get_memory_info()

            return TestResult(
                test_name=f"Stress Test ({duration_seconds}s)",
                algorithm=algorithm,
                success=True,
                execution_time_ms=execution_time,
                memory_utilization=memory_info["memory_summary"]["overall_utilization"],
                fragmentation_ratio=memory_info["memory_summary"][
                    "average_fragmentation"
                ],
                allocations_successful=successful_allocations,
                allocations_failed=failed_allocations,
                details={
                    "duration_seconds": duration_seconds,
                    "operations_performed": operations_performed,
                    "operations_per_second": operations_performed / duration_seconds,
                    "success_rate": (
                        successful_allocations
                        / (successful_allocations + failed_allocations)
                        if (successful_allocations + failed_allocations) > 0
                        else 0
                    ),
                    "active_blocks_final": len(allocated_blocks),
                },
            )

        except Exception as e:
            return TestResult(
                test_name=f"Stress Test ({duration_seconds}s)",
                algorithm=algorithm,
                success=False,
                execution_time_ms=(time.time() - start_time) * 1000,
                memory_utilization=0.0,
                fragmentation_ratio=0.0,
                allocations_successful=0,
                allocations_failed=0,
                details={},
                error_message=str(e),
            )

    def run_edge_cases_test(self, algorithm: str) -> TestResult:
        """Test edge cases and boundary conditions"""
        device = self.create_device()
        device.set_allocator(algorithm)

        start_time = time.time()
        edge_case_results = {}

        try:
            # Test 1: Zero size allocation
            try:
                result = device.allocate(size=0)
                edge_case_results["zero_size"] = {
                    "success": result.success,
                    "error": result.error_message,
                }
            except Exception as e:
                edge_case_results["zero_size"] = {"success": False, "error": str(e)}

            # Test 2: Negative size allocation
            try:
                result = device.allocate(size=-100)
                edge_case_results["negative_size"] = {
                    "success": result.success,
                    "error": result.error_message,
                }
            except Exception as e:
                edge_case_results["negative_size"] = {"success": False, "error": str(e)}

            # Test 3: Allocation larger than total memory
            try:
                result = device.allocate(size=self.device_size + 1000)
                edge_case_results["oversized"] = {
                    "success": result.success,
                    "error": result.error_message,
                }
            except Exception as e:
                edge_case_results["oversized"] = {"success": False, "error": str(e)}

            # Test 4: Maximum size allocation
            try:
                result = device.allocate(size=self.device_size)
                edge_case_results["max_size"] = {
                    "success": result.success,
                    "error": result.error_message,
                }
                if result.success:
                    device.free(result.address)
            except Exception as e:
                edge_case_results["max_size"] = {"success": False, "error": str(e)}

            # Test 5: Free invalid address
            try:
                result = device.free(address=0xDEADBEEF)
                edge_case_results["invalid_free"] = {
                    "success": result.success,
                    "error": result.error_message,
                }
            except Exception as e:
                edge_case_results["invalid_free"] = {"success": False, "error": str(e)}

            # Test 6: Double free
            try:
                alloc_result = device.allocate(size=1024)
                if alloc_result.success:
                    free_result1 = device.free(alloc_result.address)
                    free_result2 = device.free(alloc_result.address)
                    edge_case_results["double_free"] = {
                        "first_free_success": free_result1.success,
                        "second_free_success": free_result2.success,
                        "error": free_result2.error_message,
                    }
            except Exception as e:
                edge_case_results["double_free"] = {"success": False, "error": str(e)}

            # Test 7: Alignment tests (if supported)
            try:
                result = device.allocate(size=100, alignment=16)
                edge_case_results["alignment"] = {
                    "success": result.success,
                    "error": result.error_message,
                }
            except Exception as e:
                edge_case_results["alignment"] = {"success": False, "error": str(e)}

            execution_time = (time.time() - start_time) * 1000
            memory_info = device.get_memory_info()

            # Count how many edge cases were handled gracefully
            graceful_handling = sum(
                1
                for result in edge_case_results.values()
                if isinstance(result, dict) and result.get("error") is not None
            )

            return TestResult(
                test_name="Edge Cases Test",
                algorithm=algorithm,
                success=True,
                execution_time_ms=execution_time,
                memory_utilization=memory_info["memory_summary"]["overall_utilization"],
                fragmentation_ratio=memory_info["memory_summary"][
                    "average_fragmentation"
                ],
                allocations_successful=0,
                allocations_failed=0,
                details={
                    "edge_cases_tested": len(edge_case_results),
                    "graceful_handling_count": graceful_handling,
                    "edge_case_results": edge_case_results,
                },
            )

        except Exception as e:
            return TestResult(
                test_name="Edge Cases Test",
                algorithm=algorithm,
                success=False,
                execution_time_ms=(time.time() - start_time) * 1000,
                memory_utilization=0.0,
                fragmentation_ratio=0.0,
                allocations_successful=0,
                allocations_failed=0,
                details={"edge_case_results": edge_case_results},
                error_message=str(e),
            )

    def run_all_tests(
        self, test_types: Optional[List[TestType]] = None
    ) -> Dict[str, List[TestResult]]:
        """Run all tests for all algorithms"""
        if test_types is None:
            test_types = list(TestType)

        all_results = {test_type.value: [] for test_type in test_types}

        print("Starting comprehensive algorithm testing...")
        print(f"Device size: {self.device_size} bytes")
        print(f"Algorithms to test: {', '.join(self.algorithms)}")
        print(f"Test types: {', '.join([t.value for t in test_types])}")
        print("=" * 80)

        for algorithm in self.algorithms:
            print(f"\nTesting {algorithm}...")

            # Basic functionality tests
            if TestType.BASIC_FUNCTIONALITY in test_types:
                print(f"  Running basic functionality tests...")
                for pattern in self.test_patterns:
                    result = self.run_basic_functionality_test(algorithm, pattern)
                    all_results[TestType.BASIC_FUNCTIONALITY.value].append(result)
                    self.test_results.append(result)

            # Performance test
            if TestType.PERFORMANCE in test_types:
                print(f"  Running performance test...")
                result = self.run_performance_test(algorithm)
                all_results[TestType.PERFORMANCE.value].append(result)
                self.test_results.append(result)

            # Fragmentation test
            if TestType.FRAGMENTATION in test_types:
                print(f"  Running fragmentation test...")
                result = self.run_fragmentation_test(algorithm)
                all_results[TestType.FRAGMENTATION.value].append(result)
                self.test_results.append(result)

            # Stress test
            if TestType.STRESS in test_types:
                print(f"  Running stress test...")
                result = self.run_stress_test(algorithm, duration_seconds=3)
                all_results[TestType.STRESS.value].append(result)
                self.test_results.append(result)

            # Edge cases test
            if TestType.EDGE_CASES in test_types:
                print(f"  Running edge cases test...")
                result = self.run_edge_cases_test(algorithm)
                all_results[TestType.EDGE_CASES.value].append(result)
                self.test_results.append(result)

        print("\n" + "=" * 80)
        print("All tests completed!")

        return all_results

    def generate_report(self, results: Dict[str, List[TestResult]]) -> str:
        """Generate a comprehensive test report"""
        report = []
        report.append("=" * 100)
        report.append("MEMORY ALLOCATION ALGORITHMS - COMPREHENSIVE TEST REPORT")
        report.append("=" * 100)
        report.append(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Device Size: {self.device_size:,} bytes")
        report.append(f"Algorithms Tested: {', '.join(self.algorithms)}")
        report.append("")

        # Summary statistics
        report.append("SUMMARY STATISTICS")
        report.append("-" * 50)

        algorithm_stats = {
            algo: {"passed": 0, "failed": 0, "total_time": 0.0}
            for algo in self.algorithms
        }

        for test_type, test_results in results.items():
            for result in test_results:
                algorithm_stats[result.algorithm][
                    "total_time"
                ] += result.execution_time_ms
                if result.success:
                    algorithm_stats[result.algorithm]["passed"] += 1
                else:
                    algorithm_stats[result.algorithm]["failed"] += 1

        for algo, stats in algorithm_stats.items():
            total_tests = stats["passed"] + stats["failed"]
            success_rate = (
                (stats["passed"] / total_tests * 100) if total_tests > 0 else 0
            )
            report.append(
                f"{algo:15} | Tests: {total_tests:2} | Passed: {stats['passed']:2} | Failed: {stats['failed']:2} | Success: {success_rate:5.1f}% | Time: {stats['total_time']:8.1f}ms"
            )

        report.append("")

        # Detailed results by test type
        for test_type, test_results in results.items():
            if not test_results:
                continue

            report.append(f"{test_type.upper().replace('_', ' ')} RESULTS")
            report.append("-" * 50)

            # Performance comparison for this test type
            if test_type == "performance":
                report.append("Performance Metrics:")
                report.append(
                    f"{'Algorithm':<15} | {'Ops/sec':<8} | {'Avg Alloc (μs)':<12} | {'Max Alloc (μs)':<12} | {'Success Rate':<12}"
                )
                report.append("-" * 75)

                for result in test_results:
                    if result.success:
                        ops_per_sec = result.details.get("operations_per_second", 0)
                        avg_alloc = result.details.get("avg_allocation_time_us", 0)
                        max_alloc = result.details.get("max_allocation_time_us", 0)
                        success_rate = (
                            result.allocations_successful
                            / (
                                result.allocations_successful
                                + result.allocations_failed
                            )
                            * 100
                        )
                        report.append(
                            f"{result.algorithm:<15} | {ops_per_sec:8.0f} | {avg_alloc:12.2f} | {max_alloc:12.2f} | {success_rate:11.1f}%"
                        )
                    else:
                        report.append(
                            f"{result.algorithm:<15} | {'FAILED':<8} | {'N/A':<12} | {'N/A':<12} | {'0.0%':<12}"
                        )

            elif test_type == "fragmentation":
                report.append("Fragmentation Resistance:")
                report.append(
                    f"{'Algorithm':<15} | {'Resistance':<12} | {'Final Frag':<12} | {'Utilization':<12}"
                )
                report.append("-" * 60)

                for result in test_results:
                    if result.success:
                        resistance = (
                            result.details.get("fragmentation_resistance", 0) * 100
                        )
                        report.append(
                            f"{result.algorithm:<15} | {resistance:11.1f}% | {result.fragmentation_ratio:11.1f}% | {result.memory_utilization:11.1f}%"
                        )
                    else:
                        report.append(
                            f"{result.algorithm:<15} | {'FAILED':<12} | {'N/A':<12} | {'N/A':<12}"
                        )

            else:
                # Generic results table
                report.append(
                    f"{'Algorithm':<15} | {'Success':<8} | {'Time (ms)':<10} | {'Utilization':<12} | {'Fragmentation':<14}"
                )
                report.append("-" * 75)

                for result in test_results:
                    success_str = "PASS" if result.success else "FAIL"
                    report.append(
                        f"{result.algorithm:<15} | {success_str:<8} | {result.execution_time_ms:9.1f} | {result.memory_utilization:11.1f}% | {result.fragmentation_ratio:13.1f}%"
                    )

                    if not result.success and result.error_message:
                        report.append(f"    Error: {result.error_message}")

            report.append("")

        # Algorithm-specific insights
        report.append("ALGORITHM-SPECIFIC INSIGHTS")
        report.append("-" * 50)

        # Find best performer for each category
        categories = {
            "Overall Performance": ("performance", "operations_per_second"),
            "Fragmentation Resistance": ("fragmentation", "fragmentation_resistance"),
            "Memory Utilization": ("fragmentation", "memory_utilization"),
            "Fastest Allocation": ("performance", "avg_allocation_time_us"),
        }

        for category_name, (test_type, metric) in categories.items():
            if test_type in results:
                best_result = None
                best_value = float("inf") if "time" in metric else 0

                for result in results[test_type]:
                    if result.success and metric in result.details:
                        value = result.details[metric]
                        if ("time" in metric and value < best_value) or (
                            "time" not in metric and value > best_value
                        ):
                            best_value = value
                            best_result = result

                if best_result:
                    report.append(
                        f"Best {category_name}: {best_result.algorithm} ({best_value:.2f})"
                    )

        report.append("")
        report.append("RECOMMENDATIONS")
        report.append("-" * 50)
        report.append("Based on test results:")
        report.append(
            "• First Fit: Good general-purpose algorithm, fast allocation, moderate fragmentation"
        )
        report.append(
            "• Best Fit: Minimizes waste, slower allocation, good for mixed sizes"
        )
        report.append(
            "• Worst Fit: Maximizes remaining space, can help reduce fragmentation"
        )
        report.append(
            "• Next Fit: Faster than First Fit for sequential patterns, can increase fragmentation"
        )
        report.append(
            "• Buddy System: Excellent for power-of-2 sizes, predictable performance, internal fragmentation"
        )
        report.append("")

        # Detailed failure analysis
        failed_tests = [
            result
            for result_list in results.values()
            for result in result_list
            if not result.success
        ]
        if failed_tests:
            report.append("FAILURE ANALYSIS")
            report.append("-" * 50)
            for result in failed_tests:
                report.append(
                    f"• {result.algorithm} - {result.test_name}: {result.error_message}"
                )
            report.append("")

        report.append("=" * 100)
        report.append("END OF REPORT")
        report.append("=" * 100)

        return "\n".join(report)

    def save_results_to_file(
        self,
        results: Dict[str, List[TestResult]],
        filename: str = "algorithm_test_results.txt",
    ):
        """Save test results to a file"""
        report = self.generate_report(results)
        with open(filename, "w") as f:
            f.write(report)
        print(f"Test results saved to {filename}")

    def run_comparative_analysis(self) -> Dict[str, Any]:
        """Run detailed comparative analysis between algorithms"""
        print("Running comparative analysis...")

        comparison_results = {}

        # Test each algorithm with the same workload
        workload_patterns = [
            {
                "name": "Small Blocks",
                "sizes": [64] * 50,
                "description": "Many small allocations",
            },
            {
                "name": "Large Blocks",
                "sizes": [4096] * 10,
                "description": "Few large allocations",
            },
            {
                "name": "Mixed Workload",
                "sizes": [64, 128, 256, 512, 1024, 2048] * 8,
                "description": "Mixed size allocations",
            },
        ]

        for pattern in workload_patterns:
            pattern_results = {}

            for algorithm in self.algorithms:
                device = self.create_device()
                device.set_allocator(algorithm)

                start_time = time.time_ns()
                successful = 0
                failed = 0
                addresses = []

                # Allocation phase
                for size in pattern["sizes"]:
                    result = device.allocate(size=size)
                    if result.success:
                        successful += 1
                        addresses.append(result.address)
                    else:
                        failed += 1

                allocation_time = time.time_ns() - start_time

                # Free half the blocks
                free_start = time.time_ns()
                for i, addr in enumerate(addresses):
                    if i % 2 == 0:
                        device.free(addr)
                free_time = time.time_ns() - free_start

                memory_info = device.get_memory_info()
                stats = device.get_allocator_statistics()[algorithm]

                pattern_results[algorithm] = {
                    "successful_allocations": successful,
                    "failed_allocations": failed,
                    "allocation_time_ns": allocation_time,
                    "free_time_ns": free_time,
                    "final_utilization": memory_info["memory_summary"][
                        "overall_utilization"
                    ],
                    "final_fragmentation": memory_info["memory_summary"][
                        "average_fragmentation"
                    ],
                    "avg_allocation_time": stats["average_allocation_time_ns"],
                    "avg_free_time": stats["average_free_time_ns"],
                }

            comparison_results[pattern["name"]] = pattern_results

        return comparison_results

    def print_comparative_summary(self, comparison_results: Dict[str, Any]):
        """Print a summary of comparative analysis"""
        print("\n" + "=" * 80)
        print("COMPARATIVE ANALYSIS SUMMARY")
        print("=" * 80)

        for pattern_name, pattern_results in comparison_results.items():
            print(f"\n{pattern_name}:")
            print("-" * 40)

            # Find best performers
            best_speed = min(
                pattern_results.items(), key=lambda x: x[1]["allocation_time_ns"]
            )
            best_utilization = max(
                pattern_results.items(), key=lambda x: x[1]["final_utilization"]
            )
            best_fragmentation = min(
                pattern_results.items(), key=lambda x: x[1]["final_fragmentation"]
            )

            print(
                f"Fastest Allocation: {best_speed[0]} ({best_speed[1]['allocation_time_ns'] / 1000:.1f} μs total)"
            )
            print(
                f"Best Utilization: {best_utilization[0]} ({best_utilization[1]['final_utilization']:.1%})"
            )
            print(
                f"Least Fragmentation: {best_fragmentation[0]} ({best_fragmentation[1]['final_fragmentation']:.1%})"
            )

            print(f"\nDetailed Results:")
            print(
                f"{'Algorithm':<15} | {'Success':<7} | {'Util %':<6} | {'Frag %':<6} | {'Alloc Time':<12}"
            )
            print("-" * 65)

            for algo, results in pattern_results.items():
                success_rate = (
                    results["successful_allocations"]
                    / (
                        results["successful_allocations"]
                        + results["failed_allocations"]
                    )
                    * 100
                )
                print(
                    f"{algo:<15} | {success_rate:6.1f}% | {results['final_utilization']:5.1%} | {results['final_fragmentation']:5.1%} | {results['allocation_time_ns'] / 1000:11.1f} μs"
                )


def run_specific_algorithm_test(
    algorithm_name: str, test_type: TestType = TestType.BASIC_FUNCTIONALITY
):
    """Run a specific test for a single algorithm"""
    tester = AlgorithmTester()

    if algorithm_name not in tester.algorithms:
        print(f"Unknown algorithm: {algorithm_name}")
        print(f"Available algorithms: {', '.join(tester.algorithms)}")
        return

    print(f"Testing {algorithm_name} - {test_type.value}")
    print("=" * 50)

    if test_type == TestType.BASIC_FUNCTIONALITY:
        for pattern in tester.test_patterns:
            result = tester.run_basic_functionality_test(algorithm_name, pattern)
            print(
                f"{pattern.name:<20} | {'PASS' if result.success else 'FAIL':<4} | {result.execution_time_ms:6.1f}ms | Util: {result.memory_utilization:5.1%} | Frag: {result.fragmentation_ratio:5.1%}"
            )
            if not result.success:
                print(f"  Error: {result.error_message}")

    elif test_type == TestType.PERFORMANCE:
        result = tester.run_performance_test(algorithm_name)
        if result.success:
            print(
                f"Operations per second: {result.details['operations_per_second']:.0f}"
            )
            print(
                f"Average allocation time: {result.details['avg_allocation_time_us']:.2f} μs"
            )
            print(
                f"Maximum allocation time: {result.details['max_allocation_time_us']:.2f} μs"
            )
            print(
                f"Success rate: {result.allocations_successful / (result.allocations_successful + result.allocations_failed) * 100:.1f}%"
            )
        else:
            print(f"Test failed: {result.error_message}")

    elif test_type == TestType.FRAGMENTATION:
        result = tester.run_fragmentation_test(algorithm_name)
        if result.success:
            print(
                f"Fragmentation resistance: {result.details['fragmentation_resistance'] * 100:.1f}%"
            )
            print(f"Final fragmentation: {result.fragmentation_ratio:.1%}")
            print(f"Final utilization: {result.memory_utilization:.1%}")
        else:
            print(f"Test failed: {result.error_message}")

    elif test_type == TestType.STRESS:
        result = tester.run_stress_test(algorithm_name)
        if result.success:
            print(f"Operations performed: {result.details['operations_performed']}")
            print(
                f"Operations per second: {result.details['operations_per_second']:.0f}"
            )
            print(f"Success rate: {result.details['success_rate'] * 100:.1f}%")
        else:
            print(f"Test failed: {result.error_message}")

    elif test_type == TestType.EDGE_CASES:
        result = tester.run_edge_cases_test(algorithm_name)
        if result.success:
            print("Edge case handling:")
            for case, case_result in result.details["edge_case_results"].items():
                status = "HANDLED" if case_result.get("error") else "PASSED"
                print(f"  {case:<15}: {status}")
        else:
            print(f"Test failed: {result.error_message}")


def main():
    """Main function to run comprehensive tests"""
    print("Memory Allocation Algorithm Test Suite")
    print("=" * 50)

    # Create tester instance
    tester = AlgorithmTester(device_size=128 * 1024)  # 128KB device

    # Run all tests
    print("\n1. Running comprehensive test suite...")
    results = tester.run_all_tests()

    # Generate and print report
    print("\n2. Generating test report...")
    report = tester.generate_report(results)
    print(report)

    # Save results
    tester.save_results_to_file(results)

    # Run comparative analysis
    print("\n3. Running comparative analysis...")
    comparison_results = tester.run_comparative_analysis()
    tester.print_comparative_summary(comparison_results)

    # Interactive testing section
    print("\n" + "=" * 80)
    print("INTERACTIVE TESTING")
    print("=" * 80)
    print("Available commands:")
    print("- test <algorithm> <test_type>  : Run specific test")
    print("- compare                       : Run comparison")
    print("- help                          : Show this help")
    print("- quit                          : Exit")
    print("\nAlgorithms:", ", ".join(tester.algorithms))
    print("Test types:", ", ".join([t.value for t in TestType]))

    while True:
        try:
            command = input("\n> ").strip().lower()

            if command == "quit":
                break
            elif command == "help":
                print("Available commands:")
                print("- test <algorithm> <test_type>")
                print("- compare")
                print("- quit")
            elif command == "compare":
                comparison_results = tester.run_comparative_analysis()
                tester.print_comparative_summary(comparison_results)
            elif command.startswith("test "):
                parts = command.split()
                if len(parts) >= 3:
                    algo = " ".join(parts[1:-1]).title()
                    test_type_str = parts[-1]

                    # Find matching test type
                    test_type = None
                    for t in TestType:
                        if t.value == test_type_str:
                            test_type = t
                            break

                    if test_type and algo in tester.algorithms:
                        run_specific_algorithm_test(algo, test_type)
                    else:
                        print(f"Invalid algorithm or test type")
                        print(f"Algorithms: {', '.join(tester.algorithms)}")
                        print(f"Test types: {', '.join([t.value for t in TestType])}")
                else:
                    print("Usage: test <algorithm> <test_type>")
            else:
                print("Unknown command. Type 'help' for available commands.")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
