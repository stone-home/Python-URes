"""
PyTorch Model Performance Profiling Analysis System
Built on ML Visualization Framework

Analyzes execution time and memory usage from:
- torch.profiler trace data
- Memory snapshots
- Custom profiling metrics

Provides comprehensive visualization and analysis for model optimization.
"""

import polars as pl
import numpy as np
import warnings
import json
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta

# Import ML visualization framework (created above)
from ures.plot import (
    MLVisualizer,
    MLPlotConfig,
)


# ============================================================================
# PROFILING DATA STRUCTURES
# ============================================================================


@dataclass
class ProfilingMetrics:
    """Container for individual profiling metrics"""

    # Execution time metrics (milliseconds)
    total_time: float = 0.0
    cpu_time: float = 0.0
    cuda_time: float = 0.0

    # Memory metrics (bytes)
    cpu_memory_usage: float = 0.0
    gpu_memory_usage: float = 0.0
    memory_reserved: float = 0.0
    memory_allocated: float = 0.0

    # Operation details
    operation_name: str = ""
    operation_category: str = ""  # conv, linear, attention, etc.
    input_shapes: List[tuple] = field(default_factory=list)
    output_shapes: List[tuple] = field(default_factory=list)

    # Timing context
    timestamp: Optional[datetime] = None
    step: int = 0
    epoch: int = 0
    thread_id: int = 0

    # Additional metadata
    device: str = "cpu"
    is_async: bool = False


@dataclass
class LayerProfileData:
    """Detailed profiling data for individual layers"""

    layer_name: str
    layer_type: str  # Conv2d, Linear, MultiHeadAttention, etc.

    # Performance metrics
    forward_time_ms: float = 0.0
    backward_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # Memory metrics (MB)
    input_memory_mb: float = 0.0
    output_memory_mb: float = 0.0
    parameter_memory_mb: float = 0.0
    gradient_memory_mb: float = 0.0

    # Computational metrics
    flops: Optional[float] = None
    parameters: int = 0

    # Efficiency metrics
    memory_bandwidth_gbps: Optional[float] = None
    arithmetic_intensity: Optional[float] = None
    compute_utilization: Optional[float] = None

    # Layer-specific metadata
    layer_depth: int = 0
    is_training: bool = True


@dataclass
class ModelProfilingData:
    """Complete model profiling dataset"""

    # Raw profiling metrics
    operation_metrics: List[ProfilingMetrics] = field(default_factory=list)
    layer_profiles: List[LayerProfileData] = field(default_factory=list)

    # Aggregated statistics
    total_forward_time_ms: float = 0.0
    total_backward_time_ms: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    peak_cpu_memory_mb: float = 0.0

    # Model metadata
    model_name: str = ""
    model_architecture: str = ""
    batch_size: int = 1
    input_shape: tuple = ()
    total_parameters: int = 0
    device: str = "cpu"

    # Profiling metadata
    profiling_start_time: Optional[datetime] = None
    profiling_duration: Optional[timedelta] = None
    pytorch_version: str = ""
    cuda_version: str = ""


# ============================================================================
# PROFILING DATA PARSERS
# ============================================================================


class PyTorchProfilerParser:
    """Parser for PyTorch profiler output formats"""

    def __init__(self):
        self.operation_categories = {
            # Convolution operations
            "conv1d": "convolution",
            "conv2d": "convolution",
            "conv3d": "convolution",
            "conv_transpose1d": "convolution",
            "conv_transpose2d": "convolution",
            # Linear operations
            "linear": "linear",
            "addmm": "linear",
            "mm": "linear",
            "bmm": "linear",
            # Activation functions
            "relu": "activation",
            "gelu": "activation",
            "sigmoid": "activation",
            "tanh": "activation",
            "softmax": "activation",
            "log_softmax": "activation",
            # Normalization
            "batch_norm": "normalization",
            "layer_norm": "normalization",
            "group_norm": "normalization",
            "instance_norm": "normalization",
            # Pooling
            "max_pool2d": "pooling",
            "avg_pool2d": "pooling",
            "adaptive_avg_pool2d": "pooling",
            # Attention
            "scaled_dot_product_attention": "attention",
            "multi_head_attention": "attention",
            # Memory operations
            "copy_": "memory",
            "clone": "memory",
            "contiguous": "memory",
            # Gradient operations
            "backward": "gradient",
            "autograd": "gradient",
        }

    def _categorize_operation(self, op_name: str) -> str:
        """Categorize operation based on name"""
        op_lower = op_name.lower()

        for pattern, category in self.operation_categories.items():
            if pattern in op_lower:
                return category

        return "other"

    def parse_trace_file(self, trace_file: Union[str, Path]) -> ModelProfilingData:
        """
        Parse PyTorch profiler trace file (JSON format)

        Args:
                trace_file: Path to profiler trace JSON file

        Returns:
                ModelProfilingData: Parsed profiling data
        """
        try:
            with open(trace_file, "r") as f:
                trace_data = json.load(f)

            return self._parse_trace_events(trace_data)

        except Exception as e:
            warnings.warn(f"Failed to parse profiler trace: {e}")
            return ModelProfilingData()

    def _parse_trace_events(self, trace_data: Dict) -> ModelProfilingData:
        """Parse Chrome trace format events"""
        profiling_data = ModelProfilingData()

        if "traceEvents" not in trace_data:
            warnings.warn("No traceEvents found in trace data")
            return profiling_data

        events = trace_data["traceEvents"]
        operation_metrics = []

        # Parse metadata if available
        if "metadata" in trace_data:
            metadata = trace_data["metadata"]
            profiling_data.pytorch_version = metadata.get("pytorch_version", "")
            profiling_data.cuda_version = metadata.get("cuda_version", "")

        for event in events:
            try:
                # Parse complete events (duration events)
                if event.get("ph") == "X":  # Complete event
                    metric = self._parse_complete_event(event)
                    if metric:
                        operation_metrics.append(metric)

                # Parse memory events
                elif (
                    event.get("ph") == "i" and "memory" in event.get("name", "").lower()
                ):
                    self._parse_memory_event(event, profiling_data)

            except Exception as e:
                warnings.warn(f"Failed to parse event: {e}")
                continue

        profiling_data.operation_metrics = operation_metrics
        self._compute_aggregated_stats(profiling_data)

        return profiling_data

    def _parse_complete_event(self, event: Dict) -> Optional[ProfilingMetrics]:
        """Parse a complete duration event"""
        metric = ProfilingMetrics()

        # Basic timing information
        metric.operation_name = event.get("name", "unknown")
        metric.total_time = (
            event.get("dur", 0) / 1000.0
        )  # Convert microseconds to milliseconds
        metric.timestamp = datetime.fromtimestamp(event.get("ts", 0) / 1000000.0)

        # Thread and process information
        metric.thread_id = event.get("tid", 0)

        # Parse arguments for additional information
        args = event.get("args", {})

        # Extract device information
        if "device" in args:
            metric.device = str(args["device"])

        # Extract memory information
        if "memory_usage" in args:
            memory_info = args["memory_usage"]
            metric.cpu_memory_usage = memory_info.get("cpu_memory", 0)
            metric.gpu_memory_usage = memory_info.get("gpu_memory", 0)
            metric.memory_allocated = memory_info.get("allocated", 0)
            metric.memory_reserved = memory_info.get("reserved", 0)

        # Extract input/output shapes
        if "input_shapes" in args:
            metric.input_shapes = args["input_shapes"]
        if "output_shapes" in args:
            metric.output_shapes = args["output_shapes"]

        # Categorize operation
        metric.operation_category = self._categorize_operation(metric.operation_name)

        # Determine CPU vs CUDA time based on category or device
        if "cuda" in metric.device.lower() or "gpu" in metric.device.lower():
            metric.cuda_time = metric.total_time
            metric.cpu_time = 0.0
        else:
            metric.cpu_time = metric.total_time
            metric.cuda_time = 0.0

        return metric

    def _parse_memory_event(self, event: Dict, profiling_data: ModelProfilingData):
        """Parse memory usage events"""
        args = event.get("args", {})

        # Update peak memory usage
        if "allocated" in args:
            allocated_mb = args["allocated"] / (1024 * 1024)  # Convert to MB
            if "gpu" in event.get("name", "").lower():
                profiling_data.peak_gpu_memory_mb = max(
                    profiling_data.peak_gpu_memory_mb, allocated_mb
                )
            else:
                profiling_data.peak_cpu_memory_mb = max(
                    profiling_data.peak_cpu_memory_mb, allocated_mb
                )

    def _compute_aggregated_stats(self, profiling_data: ModelProfilingData):
        """Compute aggregated statistics from operation metrics"""
        if not profiling_data.operation_metrics:
            return

        # Compute total times
        profiling_data.total_forward_time_ms = sum(
            metric.total_time
            for metric in profiling_data.operation_metrics
            if "backward" not in metric.operation_name.lower()
        )

        profiling_data.total_backward_time_ms = sum(
            metric.total_time
            for metric in profiling_data.operation_metrics
            if "backward" in metric.operation_name.lower()
        )

        # Update peak memory if not set by memory events
        if profiling_data.peak_gpu_memory_mb == 0.0:
            profiling_data.peak_gpu_memory_mb = max(
                (
                    metric.gpu_memory_usage / (1024 * 1024)
                    for metric in profiling_data.operation_metrics
                ),
                default=0.0,
            )

    def parse_memory_snapshot(
        self, snapshot_file: Union[str, Path]
    ) -> ModelProfilingData:
        """
        Parse PyTorch memory snapshot file

        Args:
                snapshot_file: Path to memory snapshot (pickle format)

        Returns:
                ModelProfilingData: Memory profiling data
        """
        try:
            with open(snapshot_file, "rb") as f:
                snapshot_data = pickle.load(f)

            return self._parse_memory_snapshot_data(snapshot_data)

        except Exception as e:
            warnings.warn(f"Failed to parse memory snapshot: {e}")
            return ModelProfilingData()

    def _parse_memory_snapshot_data(self, snapshot_data: Any) -> ModelProfilingData:
        """Parse memory snapshot data structure"""
        profiling_data = ModelProfilingData()

        # Extract memory statistics
        if hasattr(snapshot_data, "device_mapping"):
            # Parse device-specific memory usage
            for device, stats in snapshot_data.device_mapping.items():
                if "cuda" in str(device).lower():
                    profiling_data.peak_gpu_memory_mb = max(
                        profiling_data.peak_gpu_memory_mb,
                        stats.get("allocated_bytes", {}).get("all", {}).get("peak", 0)
                        / (1024 * 1024),
                    )

        return profiling_data


# ============================================================================
# PROFILING DATA ANALYZER WITH ML VISUALIZATION
# ============================================================================


class PyTorchProfilingAnalyzer:
    """Comprehensive PyTorch profiling analyzer using ML visualization framework"""

    def __init__(self, profiling_data: ModelProfilingData, mode: str = "exploration"):
        """
        Initialize analyzer

        Args:
                profiling_data: Parsed profiling data
                mode: Visualization mode ("exploration" or "publication")
        """
        self.data = profiling_data
        self.visualizer = MLVisualizer(mode)
        self.mode = mode

    def to_polars_dataframes(self) -> Dict[str, pl.DataFrame]:
        """Convert profiling data to Polars DataFrames for analysis"""
        datasets = {}

        # Operations DataFrame
        if self.data.operation_metrics:
            operations_data = []

            for i, metric in enumerate(self.data.operation_metrics):
                operations_data.append(
                    {
                        "operation_id": i,
                        "operation_name": metric.operation_name,
                        "operation_category": metric.operation_category,
                        "total_time_ms": metric.total_time,
                        "cpu_time_ms": metric.cpu_time,
                        "cuda_time_ms": metric.cuda_time,
                        "cpu_memory_mb": (
                            metric.cpu_memory_usage / (1024 * 1024)
                            if metric.cpu_memory_usage
                            else 0.0
                        ),
                        "gpu_memory_mb": (
                            metric.gpu_memory_usage / (1024 * 1024)
                            if metric.gpu_memory_usage
                            else 0.0
                        ),
                        "memory_allocated_mb": (
                            metric.memory_allocated / (1024 * 1024)
                            if metric.memory_allocated
                            else 0.0
                        ),
                        "step": metric.step,
                        "epoch": metric.epoch,
                        "device": metric.device,
                        "thread_id": metric.thread_id,
                        "timestamp": (
                            metric.timestamp.isoformat() if metric.timestamp else ""
                        ),
                        "is_backward": "backward" in metric.operation_name.lower(),
                    }
                )

            datasets["operations"] = pl.DataFrame(operations_data)

        # Layer Profiles DataFrame
        if self.data.layer_profiles:
            layers_data = []

            for layer in self.data.layer_profiles:
                layers_data.append(
                    {
                        "layer_name": layer.layer_name,
                        "layer_type": layer.layer_type,
                        "forward_time_ms": layer.forward_time_ms,
                        "backward_time_ms": layer.backward_time_ms,
                        "total_time_ms": layer.total_time_ms,
                        "input_memory_mb": layer.input_memory_mb,
                        "output_memory_mb": layer.output_memory_mb,
                        "parameter_memory_mb": layer.parameter_memory_mb,
                        "gradient_memory_mb": layer.gradient_memory_mb,
                        "parameters": layer.parameters,
                        "flops": layer.flops or 0,
                        "memory_bandwidth_gbps": layer.memory_bandwidth_gbps or 0.0,
                        "arithmetic_intensity": layer.arithmetic_intensity or 0.0,
                        "compute_utilization": layer.compute_utilization or 0.0,
                        "layer_depth": layer.layer_depth,
                        "is_training": layer.is_training,
                    }
                )

            datasets["layers"] = pl.DataFrame(layers_data)

        return datasets

    # ========================================================================
    # EXECUTION TIME ANALYSIS
    # ========================================================================

    def analyze_execution_timeline(self) -> Dict[str, Any]:
        """Analyze model execution timeline and identify bottlenecks"""

        datasets = self.to_polars_dataframes()
        if "operations" not in datasets:
            return {"error": "No operations data available"}

        ops_df = datasets["operations"]
        analyses = {}

        # 1. Execution Timeline Plot
        config = MLPlotConfig(
            title="Model Execution Timeline",
            xlabel="Operation Index",
            ylabel="Execution Time (ms)",
            mode=self.mode,
        )

        # Create timeline with operation categories
        timeline_df = ops_df.with_row_count("operation_index")
        analyses["timeline_plot"] = self.visualizer.scatter_plot(
            timeline_df,
            config,
            "operation_index",
            "total_time_ms",
            color_col="operation_category",
        )

        # 2. Cumulative Time Analysis
        cumulative_df = (
            ops_df.sort("operation_id")
            .with_columns(pl.col("total_time_ms").cumsum().alias("cumulative_time_ms"))
            .with_row_count("operation_index")
        )

        config_cumulative = MLPlotConfig(
            title="Cumulative Execution Time",
            xlabel="Operation Index",
            ylabel="Cumulative Time (ms)",
            mode=self.mode,
        )

        analyses["cumulative_plot"] = self.visualizer.line_plot(
            cumulative_df, config_cumulative, "operation_index", "cumulative_time_ms"
        )

        # 3. Operation Category Distribution
        category_summary = (
            ops_df.group_by("operation_category")
            .agg(
                [
                    pl.col("total_time_ms").sum().alias("total_time"),
                    pl.col("total_time_ms").mean().alias("avg_time"),
                    pl.col("total_time_ms").count().alias("operation_count"),
                ]
            )
            .sort("total_time", descending=True)
        )

        config_category = MLPlotConfig(
            title="Time by Operation Category",
            xlabel="Operation Category",
            ylabel="Total Time (ms)",
            mode=self.mode,
        )

        analyses["category_plot"] = self.visualizer.bar_plot(
            category_summary, config_category, "operation_category", "total_time"
        )

        # 4. Time Distribution Histogram
        config_hist = MLPlotConfig(
            title="Operation Time Distribution",
            xlabel="Execution Time (ms)",
            ylabel="Frequency",
            mode=self.mode,
        )

        analyses["time_distribution"] = self.visualizer.histogram(
            ops_df, config_hist, "total_time_ms", bins=50
        )

        # 5. Bottleneck Analysis (Top 10 slowest operations)
        slowest_ops = ops_df.sort("total_time_ms", descending=True).head(10)

        config_bottleneck = MLPlotConfig(
            title="Top 10 Slowest Operations",
            xlabel="Operation Name",
            ylabel="Time (ms)",
            mode=self.mode,
        )

        analyses["bottleneck_plot"] = self.visualizer.bar_plot(
            slowest_ops, config_bottleneck, "operation_name", "total_time_ms"
        )

        # Statistical summary
        analyses["statistics"] = {
            "total_operations": len(ops_df),
            "total_time_ms": ops_df["total_time_ms"].sum(),
            "mean_time_ms": ops_df["total_time_ms"].mean(),
            "std_time_ms": ops_df["total_time_ms"].std(),
            "median_time_ms": ops_df["total_time_ms"].median(),
            "percentiles": {
                "95th": ops_df["total_time_ms"].quantile(0.95),
                "99th": ops_df["total_time_ms"].quantile(0.99),
            },
        }

        return analyses

    def analyze_layer_performance(self) -> Dict[str, Any]:
        """Analyze per-layer performance characteristics"""

        datasets = self.to_polars_dataframes()
        analyses = {}

        # Use operations data if layer data not available
        if "layers" in datasets:
            layers_df = datasets["layers"]
        elif "operations" in datasets:
            # Create pseudo-layer analysis from operations
            ops_df = datasets["operations"]
            layers_df = ops_df.group_by("operation_name").agg(
                [
                    pl.col("total_time_ms").sum().alias("total_time_ms"),
                    pl.col("total_time_ms").mean().alias("avg_time_ms"),
                    pl.col("gpu_memory_mb").max().alias("peak_memory_mb"),
                    pl.col("operation_category").first().alias("operation_category"),
                ]
            )

            # Add pseudo layer information
            layers_df = layers_df.with_columns(
                [
                    pl.col("operation_name").alias("layer_name"),
                    pl.col("operation_category").alias("layer_type"),
                    pl.lit(0.0).alias("forward_time_ms"),
                    pl.lit(0.0).alias("backward_time_ms"),
                    pl.col("peak_memory_mb").alias("input_memory_mb"),
                    pl.lit(0.0).alias("parameter_memory_mb"),
                ]
            )
        else:
            return {"error": "No layer or operation data available"}

        # 1. Layer Time Comparison
        config = MLPlotConfig(
            title="Layer Execution Time Comparison",
            xlabel="Layer Name",
            ylabel="Total Time (ms)",
            mode=self.mode,
        )

        top_layers = layers_df.sort("total_time_ms", descending=True).head(15)
        analyses["layer_comparison"] = self.visualizer.bar_plot(
            top_layers, config, "layer_name", "total_time_ms"
        )

        # 2. Forward vs Backward Time (if available)
        if (
            "forward_time_ms" in layers_df.columns
            and layers_df["forward_time_ms"].sum() > 0
        ):
            config_fwd_back = MLPlotConfig(
                title="Forward vs Backward Pass Time",
                xlabel="Forward Time (ms)",
                ylabel="Backward Time (ms)",
                mode=self.mode,
            )

            analyses["forward_backward"] = self.visualizer.scatter_plot(
                layers_df.filter(
                    (pl.col("forward_time_ms") > 0) & (pl.col("backward_time_ms") > 0)
                ),
                config_fwd_back,
                "forward_time_ms",
                "backward_time_ms",
                color_col="layer_type",
            )

        # 3. Memory vs Compute Analysis
        if "input_memory_mb" in layers_df.columns:
            memory_compute_df = layers_df.with_columns(
                (pl.col("input_memory_mb") + pl.col("parameter_memory_mb")).alias(
                    "total_memory_mb"
                )
            ).filter(pl.col("total_memory_mb") > 0)

            if len(memory_compute_df) > 0:
                config_mem_comp = MLPlotConfig(
                    title="Memory Usage vs Execution Time",
                    xlabel="Total Memory (MB)",
                    ylabel="Execution Time (ms)",
                    mode=self.mode,
                )

                analyses["memory_compute"] = self.visualizer.scatter_plot(
                    memory_compute_df,
                    config_mem_comp,
                    "total_memory_mb",
                    "total_time_ms",
                    color_col="layer_type",
                )

        # 4. Layer Type Summary
        if "layer_type" in layers_df.columns:
            type_summary = (
                layers_df.group_by("layer_type")
                .agg(
                    [
                        pl.col("total_time_ms").sum().alias("total_time"),
                        pl.col("total_time_ms").count().alias("layer_count"),
                        pl.col("total_time_ms").mean().alias("avg_time"),
                    ]
                )
                .sort("total_time", descending=True)
            )

            config_type = MLPlotConfig(
                title="Performance by Layer Type",
                xlabel="Layer Type",
                ylabel="Total Time (ms)",
                mode=self.mode,
            )

            analyses["layer_type_summary"] = self.visualizer.bar_plot(
                type_summary, config_type, "layer_type", "total_time"
            )

            analyses["type_statistics"] = type_summary.to_dict(as_series=False)

        return analyses

    # ========================================================================
    # MEMORY ANALYSIS
    # ========================================================================

    def analyze_memory_usage(self) -> Dict[str, Any]:
        """Comprehensive memory usage analysis"""

        datasets = self.to_polars_dataframes()
        if "operations" not in datasets:
            return {"error": "No operations data available"}

        ops_df = datasets["operations"]
        analyses = {}

        # Filter to operations with memory data
        memory_ops = ops_df.filter(
            (pl.col("gpu_memory_mb") > 0) | (pl.col("cpu_memory_mb") > 0)
        )

        if len(memory_ops) == 0:
            return {"warning": "No memory usage data found in profiling"}

        # 1. Memory Usage Timeline
        config_timeline = MLPlotConfig(
            title="GPU Memory Usage Timeline",
            xlabel="Operation Index",
            ylabel="Memory Usage (MB)",
            mode=self.mode,
        )

        memory_timeline = memory_ops.with_row_count("operation_index")
        analyses["memory_timeline"] = self.visualizer.line_plot(
            memory_timeline, config_timeline, "operation_index", "gpu_memory_mb"
        )

        # 2. Memory by Operation Category
        memory_by_category = (
            memory_ops.group_by("operation_category")
            .agg(
                [
                    pl.col("gpu_memory_mb").max().alias("peak_memory"),
                    pl.col("gpu_memory_mb").mean().alias("avg_memory"),
                    pl.col("gpu_memory_mb").count().alias("operation_count"),
                ]
            )
            .sort("peak_memory", descending=True)
        )

        config_category = MLPlotConfig(
            title="Peak Memory by Operation Category",
            xlabel="Operation Category",
            ylabel="Peak Memory (MB)",
            mode=self.mode,
        )

        analyses["memory_by_category"] = self.visualizer.bar_plot(
            memory_by_category, config_category, "operation_category", "peak_memory"
        )

        # 3. Memory vs Time Scatter
        config_scatter = MLPlotConfig(
            title="Memory Usage vs Execution Time",
            xlabel="Execution Time (ms)",
            ylabel="GPU Memory (MB)",
            mode=self.mode,
        )

        analyses["memory_time_scatter"] = self.visualizer.scatter_plot(
            memory_ops,
            config_scatter,
            "total_time_ms",
            "gpu_memory_mb",
            color_col="operation_category",
        )

        # 4. Memory Distribution
        config_hist = MLPlotConfig(
            title="Memory Usage Distribution",
            xlabel="GPU Memory (MB)",
            ylabel="Frequency",
            mode=self.mode,
        )

        analyses["memory_distribution"] = self.visualizer.histogram(
            memory_ops, config_hist, "gpu_memory_mb", bins=30
        )

        # 5. Memory Growth Analysis
        memory_growth = (
            memory_ops.sort("operation_id")
            .with_columns(
                [
                    pl.col("gpu_memory_mb").cummax().alias("cumulative_peak_memory"),
                    pl.col("gpu_memory_mb").diff().alias("memory_delta"),
                ]
            )
            .with_row_count("operation_index")
        )

        config_growth = MLPlotConfig(
            title="Memory Growth Over Time",
            xlabel="Operation Index",
            ylabel="Cumulative Peak Memory (MB)",
            mode=self.mode,
        )

        analyses["memory_growth"] = self.visualizer.line_plot(
            memory_growth, config_growth, "operation_index", "cumulative_peak_memory"
        )

        # Memory statistics
        analyses["memory_statistics"] = {
            "peak_gpu_memory_mb": memory_ops["gpu_memory_mb"].max(),
            "avg_gpu_memory_mb": memory_ops["gpu_memory_mb"].mean(),
            "memory_volatility": memory_ops["gpu_memory_mb"].std(),
            "total_operations_with_memory": len(memory_ops),
            "memory_growth_rate": self._calculate_memory_growth_rate(memory_ops),
        }

        return analyses

    def _calculate_memory_growth_rate(self, df: pl.DataFrame) -> float:
        """Calculate memory growth rate over operations"""
        if len(df) < 2:
            return 0.0

        # Simple linear trend
        indices = np.arange(len(df))
        memory = df["gpu_memory_mb"].to_numpy()

        # Remove zeros for better trend calculation
        non_zero_mask = memory > 0
        if np.sum(non_zero_mask) < 2:
            return 0.0

        indices_clean = indices[non_zero_mask]
        memory_clean = memory[non_zero_mask]

        # Linear regression slope
        if len(indices_clean) > 1:
            slope = np.polyfit(indices_clean, memory_clean, 1)[0]
            return float(slope)

        return 0.0

    # ========================================================================
    # BOTTLENECK IDENTIFICATION
    # ========================================================================

    def identify_bottlenecks(self) -> Dict[str, Any]:
        """Comprehensive bottleneck identification and optimization suggestions"""

        datasets = self.to_polars_dataframes()
        if "operations" not in datasets:
            return {"error": "No operations data available"}

        ops_df = datasets["operations"]
        bottlenecks = {
            "compute_bottlenecks": {},
            "memory_bottlenecks": {},
            "io_bottlenecks": {},
            "optimization_suggestions": [],
        }

        # 1. Compute Bottlenecks - Operations consuming >5% of total time
        total_time = ops_df["total_time_ms"].sum()
        time_threshold = total_time * 0.05

        compute_bottlenecks = ops_df.filter(
            pl.col("total_time_ms") > time_threshold
        ).sort("total_time_ms", descending=True)

        if len(compute_bottlenecks) > 0:
            bottlenecks["compute_bottlenecks"] = {
                "operations": compute_bottlenecks.to_dict(as_series=False),
                "total_time_percentage": compute_bottlenecks["total_time_ms"].sum()
                / total_time
                * 100,
                "operation_count": len(compute_bottlenecks),
            }

            # Visualization
            config = MLPlotConfig(
                title="Compute Bottlenecks (>5% of total time)",
                xlabel="Operation Name",
                ylabel="Time (ms)",
                mode=self.mode,
            )

            bottlenecks["compute_bottleneck_plot"] = self.visualizer.bar_plot(
                compute_bottlenecks.head(10), config, "operation_name", "total_time_ms"
            )

        # 2. Memory Bottlenecks - Operations using >10% of peak memory
        memory_ops = ops_df.filter(pl.col("gpu_memory_mb") > 0)

        if len(memory_ops) > 0:
            peak_memory = memory_ops["gpu_memory_mb"].max()
            memory_threshold = peak_memory * 0.1

            memory_bottlenecks = memory_ops.filter(
                pl.col("gpu_memory_mb") > memory_threshold
            ).sort("gpu_memory_mb", descending=True)

            if len(memory_bottlenecks) > 0:
                bottlenecks["memory_bottlenecks"] = {
                    "operations": memory_bottlenecks.to_dict(as_series=False),
                    "peak_memory_mb": peak_memory,
                    "operation_count": len(memory_bottlenecks),
                }

                # Memory bottleneck visualization
                config_mem = MLPlotConfig(
                    title="Memory Bottlenecks (>10% of peak memory)",
                    xlabel="Operation Name",
                    ylabel="Memory (MB)",
                    mode=self.mode,
                )

                bottlenecks["memory_bottleneck_plot"] = self.visualizer.bar_plot(
                    memory_bottlenecks.head(10),
                    config_mem,
                    "operation_name",
                    "gpu_memory_mb",
                )

        # 3. I/O and Data Movement Bottlenecks
        io_operations = ops_df.filter(
            pl.col("operation_category").is_in(["memory", "copy", "communication"])
        )

        if len(io_operations) > 0:
            io_time_percentage = io_operations["total_time_ms"].sum() / total_time * 100

            bottlenecks["io_bottlenecks"] = {
                "operations": io_operations.sort(
                    "total_time_ms", descending=True
                ).to_dict(as_series=False),
                "total_time_percentage": io_time_percentage,
                "operation_count": len(io_operations),
            }

        # 4. Generate Optimization Suggestions
        suggestions = []

        # Compute optimization suggestions
        if "compute_bottlenecks" in bottlenecks and bottlenecks["compute_bottlenecks"]:
            compute_pct = bottlenecks["compute_bottlenecks"]["total_time_percentage"]

            if compute_pct > 70:
                suggestions.append(
                    {
                        "type": "compute",
                        "priority": "high",
                        "issue": f"Compute bottlenecks account for {compute_pct:.1f}% of execution time",
                        "suggestion": "Consider kernel fusion, mixed precision, or model architecture optimization",
                        "affected_operations": bottlenecks["compute_bottlenecks"][
                            "operations"
                        ]["operation_name"][:3],
                    }
                )
            elif compute_pct > 40:
                suggestions.append(
                    {
                        "type": "compute",
                        "priority": "medium",
                        "issue": f"Moderate compute bottlenecks ({compute_pct:.1f}% of time)",
                        "suggestion": "Profile individual operations for optimization opportunities",
                        "affected_operations": bottlenecks["compute_bottlenecks"][
                            "operations"
                        ]["operation_name"][:3],
                    }
                )

        # Memory optimization suggestions
        if "memory_bottlenecks" in bottlenecks and bottlenecks["memory_bottlenecks"]:
            peak_memory = bottlenecks["memory_bottlenecks"]["peak_memory_mb"]

            if peak_memory > 16000:  # 16GB threshold
                suggestions.append(
                    {
                        "type": "memory",
                        "priority": "high",
                        "issue": f"Very high memory usage: {peak_memory:.0f} MB",
                        "suggestion": "Consider gradient checkpointing, model sharding, or reducing batch size",
                        "peak_memory_mb": peak_memory,
                    }
                )
            elif peak_memory > 8000:  # 8GB threshold
                suggestions.append(
                    {
                        "type": "memory",
                        "priority": "medium",
                        "issue": f"High memory usage: {peak_memory:.0f} MB",
                        "suggestion": "Monitor memory growth and consider optimization techniques",
                        "peak_memory_mb": peak_memory,
                    }
                )

        # I/O optimization suggestions
        if "io_bottlenecks" in bottlenecks and bottlenecks["io_bottlenecks"]:
            io_pct = bottlenecks["io_bottlenecks"]["total_time_percentage"]

            if io_pct > 20:
                suggestions.append(
                    {
                        "type": "io",
                        "priority": "medium",
                        "issue": f"I/O operations consume {io_pct:.1f}% of execution time",
                        "suggestion": "Consider data prefetching, async operations, or memory pinning",
                        "io_time_percentage": io_pct,
                    }
                )

        bottlenecks["optimization_suggestions"] = suggestions

        # Create bottleneck summary visualization
        if len(suggestions) > 0:
            bottleneck_summary = pl.DataFrame(
                {
                    "bottleneck_type": [s["type"] for s in suggestions],
                    "priority_score": [
                        (
                            3
                            if s["priority"] == "high"
                            else 2 if s["priority"] == "medium" else 1
                        )
                        for s in suggestions
                    ],
                }
            )

            config_summary = MLPlotConfig(
                title="Bottleneck Priority Summary",
                xlabel="Bottleneck Type",
                ylabel="Priority Score",
                mode=self.mode,
            )

            bottlenecks["bottleneck_summary_plot"] = self.visualizer.bar_plot(
                bottleneck_summary, config_summary, "bottleneck_type", "priority_score"
            )

        return bottlenecks

    # ========================================================================
    # COMPARATIVE ANALYSIS
    # ========================================================================

    def compare_with_baseline(
        self, baseline_analyzer: "PyTorchProfilingAnalyzer"
    ) -> Dict[str, Any]:
        """Compare current profiling results with baseline"""

        comparison = {
            "execution_time_comparison": None,
            "memory_comparison": None,
            "bottleneck_comparison": {},
            "performance_improvements": {},
            "recommendations": [],
        }

        # Get datasets from both analyzers
        current_datasets = self.to_polars_dataframes()
        baseline_datasets = baseline_analyzer.to_polars_dataframes()

        if (
            "operations" not in current_datasets
            or "operations" not in baseline_datasets
        ):
            return {"error": "Insufficient data for comparison"}

        current_ops = current_datasets["operations"]
        baseline_ops = baseline_datasets["operations"]

        # 1. Overall Performance Comparison
        current_total_time = current_ops["total_time_ms"].sum()
        baseline_total_time = baseline_ops["total_time_ms"].sum()
        time_improvement = (
            (baseline_total_time - current_total_time) / baseline_total_time * 100
        )

        current_peak_memory = current_ops["gpu_memory_mb"].max()
        baseline_peak_memory = baseline_ops["gpu_memory_mb"].max()
        memory_improvement = (
            (baseline_peak_memory - current_peak_memory) / baseline_peak_memory * 100
        )

        comparison["performance_improvements"] = {
            "execution_time_improvement_pct": time_improvement,
            "memory_improvement_pct": memory_improvement,
            "current_total_time_ms": current_total_time,
            "baseline_total_time_ms": baseline_total_time,
            "current_peak_memory_mb": current_peak_memory,
            "baseline_peak_memory_mb": baseline_peak_memory,
        }

        # 2. Category-wise Comparison
        current_category = current_ops.group_by("operation_category").agg(
            pl.col("total_time_ms").sum().alias("current_time")
        )

        baseline_category = baseline_ops.group_by("operation_category").agg(
            pl.col("total_time_ms").sum().alias("baseline_time")
        )

        category_comparison = (
            current_category.join(
                baseline_category, on="operation_category", how="outer"
            )
            .with_columns(
                [
                    pl.col("current_time").fill_null(0),
                    pl.col("baseline_time").fill_null(0),
                ]
            )
            .with_columns(
                (
                    (pl.col("baseline_time") - pl.col("current_time"))
                    / pl.col("baseline_time")
                    * 100
                ).alias("improvement_pct")
            )
        )

        config_comparison = MLPlotConfig(
            title="Performance Comparison by Category",
            xlabel="Current Time (ms)",
            ylabel="Baseline Time (ms)",
            mode=self.mode,
        )

        comparison["execution_time_comparison"] = self.visualizer.scatter_plot(
            category_comparison.filter(
                (pl.col("current_time") > 0) & (pl.col("baseline_time") > 0)
            ),
            config_comparison,
            "current_time",
            "baseline_time",
            color_col="operation_category",
        )

        # 3. Memory Comparison Visualization
        memory_comparison_data = pl.DataFrame(
            {
                "configuration": ["Baseline", "Current"],
                "peak_memory_mb": [baseline_peak_memory, current_peak_memory],
                "improvement": [0, memory_improvement],
            }
        )

        config_memory = MLPlotConfig(
            title="Peak Memory Usage Comparison",
            xlabel="Configuration",
            ylabel="Peak Memory (MB)",
            mode=self.mode,
        )

        comparison["memory_comparison"] = self.visualizer.bar_plot(
            memory_comparison_data, config_memory, "configuration", "peak_memory_mb"
        )

        # 4. Generate Comparison Recommendations
        recommendations = []

        if time_improvement > 10:
            recommendations.append(
                {
                    "type": "performance",
                    "message": f"Excellent performance improvement: {time_improvement:.1f}% faster execution",
                    "impact": "high",
                }
            )
        elif time_improvement > 0:
            recommendations.append(
                {
                    "type": "performance",
                    "message": f"Moderate performance improvement: {time_improvement:.1f}% faster execution",
                    "impact": "medium",
                }
            )
        elif time_improvement < -10:
            recommendations.append(
                {
                    "type": "performance",
                    "message": f"Performance regression: {abs(time_improvement):.1f}% slower execution",
                    "impact": "high",
                    "action": "Review recent changes and consider rolling back",
                }
            )

        if memory_improvement > 20:
            recommendations.append(
                {
                    "type": "memory",
                    "message": f"Significant memory optimization: {memory_improvement:.1f}% reduction",
                    "impact": "high",
                }
            )
        elif memory_improvement < -20:
            recommendations.append(
                {
                    "type": "memory",
                    "message": f"Memory usage increased: {abs(memory_improvement):.1f}% higher",
                    "impact": "medium",
                    "action": "Consider memory optimization techniques",
                }
            )

        comparison["recommendations"] = recommendations

        return comparison

    # ========================================================================
    # COMPREHENSIVE DASHBOARD GENERATION
    # ========================================================================

    def generate_profiling_dashboard(
        self, output_dir: Union[str, Path]
    ) -> Dict[str, Any]:
        """Generate comprehensive profiling analysis dashboard"""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("🔬 Generating PyTorch Profiling Dashboard...")

        dashboard = {
            "execution_analysis": {},
            "memory_analysis": {},
            "layer_analysis": {},
            "bottleneck_analysis": {},
            "summary_statistics": {},
            "generated_plots": [],
            "analysis_metadata": {},
        }

        # 1. Execution Time Analysis
        print("⏱️  Analyzing execution timeline...")
        exec_analysis = self.analyze_execution_timeline()
        dashboard["execution_analysis"] = exec_analysis

        # Save execution plots
        for plot_name, fig in exec_analysis.items():
            if fig is not None and hasattr(fig, "savefig"):
                plot_path = output_dir / f"execution_{plot_name}.png"
                self.visualizer.save_figure(
                    fig, plot_path, MLPlotConfig(mode=self.mode)
                )
                dashboard["generated_plots"].append(str(plot_path))
            elif fig is not None:
                # Handle HoloViews plots
                plot_path = output_dir / f"execution_{plot_name}.html"
                try:
                    import holoviews as hv

                    hv.save(fig, str(plot_path))
                    dashboard["generated_plots"].append(str(plot_path))
                except:
                    pass

        # 2. Memory Usage Analysis
        print("💾 Analyzing memory usage...")
        memory_analysis = self.analyze_memory_usage()
        dashboard["memory_analysis"] = memory_analysis

        # Save memory plots
        for plot_name, fig in memory_analysis.items():
            if fig is not None and hasattr(fig, "savefig"):
                plot_path = output_dir / f"memory_{plot_name}.png"
                self.visualizer.save_figure(
                    fig, plot_path, MLPlotConfig(mode=self.mode)
                )
                dashboard["generated_plots"].append(str(plot_path))
            elif fig is not None:
                plot_path = output_dir / f"memory_{plot_name}.html"
                try:
                    import holoviews as hv

                    hv.save(fig, str(plot_path))
                    dashboard["generated_plots"].append(str(plot_path))
                except:
                    pass

        # 3. Layer Performance Analysis
        print("🔍 Analyzing layer performance...")
        layer_analysis = self.analyze_layer_performance()
        dashboard["layer_analysis"] = layer_analysis

        # Save layer plots
        for plot_name, fig in layer_analysis.items():
            if fig is not None and hasattr(fig, "savefig"):
                plot_path = output_dir / f"layer_{plot_name}.png"
                self.visualizer.save_figure(
                    fig, plot_path, MLPlotConfig(mode=self.mode)
                )
                dashboard["generated_plots"].append(str(plot_path))
            elif fig is not None:
                plot_path = output_dir / f"layer_{plot_name}.html"
                try:
                    import holoviews as hv

                    hv.save(fig, str(plot_path))
                    dashboard["generated_plots"].append(str(plot_path))
                except:
                    pass

        # 4. Bottleneck Analysis
        print("🚨 Identifying bottlenecks...")
        bottleneck_analysis = self.identify_bottlenecks()
        dashboard["bottleneck_analysis"] = bottleneck_analysis

        # Save bottleneck plots
        for plot_name, fig in bottleneck_analysis.items():
            if plot_name.endswith("_plot") and fig is not None:
                if hasattr(fig, "savefig"):
                    plot_path = output_dir / f"bottleneck_{plot_name}.png"
                    self.visualizer.save_figure(
                        fig, plot_path, MLPlotConfig(mode=self.mode)
                    )
                    dashboard["generated_plots"].append(str(plot_path))
                else:
                    plot_path = output_dir / f"bottleneck_{plot_name}.html"
                    try:
                        import holoviews as hv

                        hv.save(fig, str(plot_path))
                        dashboard["generated_plots"].append(str(plot_path))
                    except:
                        pass

        # 5. Generate Summary Statistics
        print("📊 Computing summary statistics...")
        summary_stats = self._generate_comprehensive_summary()
        dashboard["summary_statistics"] = summary_stats

        # 6. Create Analysis Metadata
        dashboard["analysis_metadata"] = {
            "model_name": self.data.model_name,
            "batch_size": self.data.batch_size,
            "device": self.data.device,
            "total_parameters": self.data.total_parameters,
            "pytorch_version": self.data.pytorch_version,
            "analysis_timestamp": datetime.now().isoformat(),
            "analysis_mode": self.mode,
            "backend_info": self.visualizer.get_backend_info(),
            "total_plots_generated": len(dashboard["generated_plots"]),
        }

        # 7. Save dashboard metadata as JSON
        metadata_path = output_dir / "profiling_dashboard_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(dashboard["analysis_metadata"], f, indent=2, default=str)

        # 8. Generate HTML dashboard report
        self._generate_html_report(dashboard, output_dir)

        print("✅ Profiling Dashboard Generation Complete!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📊 Total plots generated: {len(dashboard['generated_plots'])}")
        print(f"📈 Analysis mode: {self.mode}")

        return dashboard

    def _generate_comprehensive_summary(self) -> Dict[str, Any]:
        """Generate comprehensive summary statistics"""

        datasets = self.to_polars_dataframes()
        summary = {
            "model_statistics": {},
            "execution_statistics": {},
            "memory_statistics": {},
            "performance_insights": {},
        }

        # Model Statistics
        summary["model_statistics"] = {
            "model_name": self.data.model_name,
            "total_parameters": self.data.total_parameters,
            "batch_size": self.data.batch_size,
            "input_shape": self.data.input_shape,
            "device": self.data.device,
        }

        # Execution Statistics
        if "operations" in datasets:
            ops_df = datasets["operations"]

            summary["execution_statistics"] = {
                "total_operations": len(ops_df),
                "total_execution_time_ms": ops_df["total_time_ms"].sum(),
                "average_operation_time_ms": ops_df["total_time_ms"].mean(),
                "median_operation_time_ms": ops_df["total_time_ms"].median(),
                "execution_time_std_ms": ops_df["total_time_ms"].std(),
                "slowest_operation_time_ms": ops_df["total_time_ms"].max(),
                "fastest_operation_time_ms": ops_df["total_time_ms"].min(),
                "forward_time_ms": self.data.total_forward_time_ms,
                "backward_time_ms": self.data.total_backward_time_ms,
                "forward_backward_ratio": (
                    self.data.total_forward_time_ms
                    / (self.data.total_backward_time_ms + 1e-8)
                ),
            }

            # Operation category breakdown
            category_stats = (
                ops_df.group_by("operation_category")
                .agg(
                    [
                        pl.col("total_time_ms").sum().alias("total_time"),
                        pl.col("total_time_ms").count().alias("operation_count"),
                    ]
                )
                .sort("total_time", descending=True)
            )

            summary["execution_statistics"]["category_breakdown"] = (
                category_stats.to_dict(as_series=False)
            )

        # Memory Statistics
        if "operations" in datasets:
            ops_df = datasets["operations"]
            memory_ops = ops_df.filter(pl.col("gpu_memory_mb") > 0)

            if len(memory_ops) > 0:
                summary["memory_statistics"] = {
                    "peak_gpu_memory_mb": memory_ops["gpu_memory_mb"].max(),
                    "average_gpu_memory_mb": memory_ops["gpu_memory_mb"].mean(),
                    "memory_volatility": memory_ops["gpu_memory_mb"].std(),
                    "operations_with_memory_data": len(memory_ops),
                    "memory_efficiency": (
                        memory_ops["gpu_memory_mb"].mean()
                        / (memory_ops["gpu_memory_mb"].max() + 1e-8)
                    ),
                    "memory_growth_rate": self._calculate_memory_growth_rate(
                        memory_ops
                    ),
                }

        # Performance Insights
        if "operations" in datasets:
            ops_df = datasets["operations"]
            total_time = ops_df["total_time_ms"].sum()

            # Find dominant operation categories
            category_times = (
                ops_df.group_by("operation_category")
                .agg(pl.col("total_time_ms").sum().alias("total_time"))
                .sort("total_time", descending=True)
            )

            dominant_category = category_times.row(0)
            dominant_percentage = (dominant_category[1] / total_time) * 100

            summary["performance_insights"] = {
                "dominant_operation_category": dominant_category[0],
                "dominant_category_percentage": dominant_percentage,
                "operations_over_10ms": len(
                    ops_df.filter(pl.col("total_time_ms") > 10)
                ),
                "operations_over_100ms": len(
                    ops_df.filter(pl.col("total_time_ms") > 100)
                ),
                "long_tail_percentage": (
                    ops_df.filter(
                        pl.col("total_time_ms") < ops_df["total_time_ms"].quantile(0.9)
                    )["total_time_ms"].sum()
                    / total_time
                    * 100
                ),
            }

        return summary

    def _generate_html_report(self, dashboard: Dict[str, Any], output_dir: Path):
        """Generate HTML dashboard report"""

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>PyTorch Profiling Dashboard - {self.data.model_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
                .plot {{ margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>PyTorch Model Profiling Dashboard</h1>
                <p><strong>Model:</strong> {self.data.model_name}</p>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Analysis Mode:</strong> {self.mode}</p>
            </div>

            <div class="section">
                <h2>Model Overview</h2>
                <div class="metric"><strong>Batch Size:</strong> {self.data.batch_size}</div>
                <div class="metric"><strong>Device:</strong> {self.data.device}</div>
                <div class="metric"><strong>Parameters:</strong> {self.data.total_parameters:,}</div>
                <div class="metric"><strong>Peak Memory:</strong> {self.data.peak_gpu_memory_mb:.1f} MB</div>
            </div>

            <div class="section">
                <h2>Performance Summary</h2>
        """

        if (
            "summary_statistics" in dashboard
            and "execution_statistics" in dashboard["summary_statistics"]
        ):
            exec_stats = dashboard["summary_statistics"]["execution_statistics"]
            html_content += f"""
                <div class="metric"><strong>Total Time:</strong> {exec_stats.get('total_execution_time_ms', 0):.2f} ms</div>
                <div class="metric"><strong>Total Operations:</strong> {exec_stats.get('total_operations', 0):,}</div>
                <div class="metric"><strong>Avg Operation Time:</strong> {exec_stats.get('average_operation_time_ms', 0):.3f} ms</div>
                <div class="metric"><strong>Forward/Backward Ratio:</strong> {exec_stats.get('forward_backward_ratio', 0):.2f}</div>
            """

        html_content += """
            </div>

            <div class="section">
                <h2>Optimization Recommendations</h2>
        """

        if (
            "bottleneck_analysis" in dashboard
            and "optimization_suggestions" in dashboard["bottleneck_analysis"]
        ):
            suggestions = dashboard["bottleneck_analysis"]["optimization_suggestions"]
            if suggestions:
                html_content += "<ul>"
                for suggestion in suggestions:
                    priority_color = {
                        "high": "#ff6b6b",
                        "medium": "#ffd93d",
                        "low": "#6bcf7f",
                    }.get(suggestion["priority"], "#ddd")
                    html_content += f"""
                        <li style="margin: 10px 0; padding: 10px; border-left: 4px solid {priority_color};">
                            <strong>{suggestion['type'].title()} ({suggestion['priority'].title()} Priority):</strong><br>
                            {suggestion['issue']}<br>
                            <em>Suggestion: {suggestion['suggestion']}</em>
                        </li>
                    """
                html_content += "</ul>"
            else:
                html_content += "<p>No critical bottlenecks identified. Model performance appears optimal.</p>"

        html_content += f"""
            </div>

            <div class="section">
                <h2>Generated Visualizations</h2>
                <p>Total plots generated: {len(dashboard['generated_plots'])}</p>
                <ul>
        """

        for plot_path in dashboard["generated_plots"]:
            plot_name = Path(plot_path).name
            html_content += f'<li><a href="{plot_name}">{plot_name}</a></li>'

        html_content += """
                </ul>
            </div>

            <div class="section">
                <h2>Analysis Details</h2>
                <p>For detailed analysis, please refer to the individual plot files and the metadata JSON file.</p>
                <p>This dashboard was generated using the PyTorch Profiling Analysis Framework built on the ML Visualization System.</p>
            </div>
        </body>
        </html>
        """

        # Save HTML report
        html_path = output_dir / "profiling_dashboard_report.html"
        with open(html_path, "w") as f:
            f.write(html_content)

        dashboard["html_report_path"] = str(html_path)


# ============================================================================
# CONVENIENCE FUNCTIONS AND TEMPLATES
# ============================================================================


def analyze_pytorch_trace(
    trace_file: Union[str, Path],
    output_dir: Union[str, Path] = "profiling_analysis",
    mode: str = "exploration",
) -> Dict[str, Any]:
    """
    Complete analysis of PyTorch profiler trace file

    Args:
            trace_file: Path to PyTorch profiler trace JSON file
            output_dir: Directory to save analysis results
            mode: Visualization mode ("exploration" or "publication")

    Returns:
            Dict containing comprehensive analysis results
    """

    print(f"🔬 Starting PyTorch Profiler Analysis")
    print(f"📁 Input file: {trace_file}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎨 Visualization mode: {mode}")

    # Parse profiler data
    parser = PyTorchProfilerParser()
    profiling_data = parser.parse_trace_file(trace_file)

    if not profiling_data.operation_metrics:
        print("❌ No profiling data found in trace file")
        return {"error": "No profiling data available"}

    print(f"✅ Parsed {len(profiling_data.operation_metrics)} operations")

    # Create analyzer with specified mode
    analyzer = PyTorchProfilingAnalyzer(profiling_data, mode=mode)

    # Generate comprehensive dashboard
    dashboard = analyzer.generate_profiling_dashboard(output_dir)

    print("🎯 Analysis Summary:")
    if "summary_statistics" in dashboard:
        stats = dashboard["summary_statistics"]
        if "execution_statistics" in stats:
            exec_stats = stats["execution_statistics"]
            print(
                f"   • Total execution time: {exec_stats.get('total_execution_time_ms', 0):.2f} ms"
            )
            print(
                f"   • Operations analyzed: {exec_stats.get('total_operations', 0):,}"
            )
            print(
                f"   • Average operation time: {exec_stats.get('average_operation_time_ms', 0):.3f} ms"
            )

        if "memory_statistics" in stats:
            mem_stats = stats["memory_statistics"]
            print(
                f"   • Peak GPU memory: {mem_stats.get('peak_gpu_memory_mb', 0):.1f} MB"
            )

    return dashboard


def analyze_memory_snapshot(
    snapshot_file: Union[str, Path],
    output_dir: Union[str, Path] = "memory_analysis",
    mode: str = "publication",
) -> Dict[str, Any]:
    """
    Analyze PyTorch memory snapshot

    Args:
            snapshot_file: Path to memory snapshot pickle file
            output_dir: Directory to save analysis results
            mode: Visualization mode ("exploration" or "publication")

    Returns:
            Dict containing memory analysis results
    """

    print("💾 Starting Memory Snapshot Analysis")

    parser = PyTorchProfilerParser()
    profiling_data = parser.parse_memory_snapshot(snapshot_file)

    analyzer = PyTorchProfilingAnalyzer(profiling_data, mode=mode)
    memory_analysis = analyzer.analyze_memory_usage()

    print(f"📊 Memory Analysis Complete")
    print(f"   • Peak memory analyzed: {profiling_data.peak_gpu_memory_mb:.1f} MB")

    return memory_analysis


def compare_profiling_results(
    baseline_trace: Union[str, Path],
    optimized_trace: Union[str, Path],
    output_dir: Union[str, Path] = "comparison_analysis",
    mode: str = "publication",
) -> Dict[str, Any]:
    """
    Compare performance between two profiling traces

    Args:
            baseline_trace: Path to baseline profiler trace
            optimized_trace: Path to optimized profiler trace
            output_dir: Directory to save comparison results
            mode: Visualization mode ("exploration" or "publication")

    Returns:
            Dict containing comparison analysis
    """

    print("🔄 Starting Profiling Results Comparison")

    # Parse both configurations
    parser = PyTorchProfilerParser()

    baseline_data = parser.parse_trace_file(baseline_trace)
    optimized_data = parser.parse_trace_file(optimized_trace)

    if not baseline_data.operation_metrics or not optimized_data.operation_metrics:
        return {"error": "Insufficient data for comparison"}

    baseline_analyzer = PyTorchProfilingAnalyzer(baseline_data, mode=mode)
    optimized_analyzer = PyTorchProfilingAnalyzer(optimized_data, mode=mode)

    # Perform comparison
    comparison = optimized_analyzer.compare_with_baseline(baseline_analyzer)

    # Save comparison visualizations
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for plot_name, fig in comparison.items():
        if plot_name.endswith("_plot") or plot_name.endswith("_comparison"):
            if fig is not None:
                plot_path = output_dir / f"comparison_{plot_name}.png"
                try:
                    if hasattr(fig, "savefig"):
                        optimized_analyzer.visualizer.save_figure(
                            fig, plot_path, MLPlotConfig(mode=mode)
                        )
                    else:
                        # Handle HoloViews plots
                        import holoviews as hv

                        hv.save(fig, str(plot_path.with_suffix(".html")))
                except Exception as e:
                    warnings.warn(f"Failed to save comparison plot: {e}")

    print("✅ Comparison Analysis Complete")
    if "performance_improvements" in comparison:
        improvements = comparison["performance_improvements"]
        time_improvement = improvements.get("execution_time_improvement_pct", 0)
        memory_improvement = improvements.get("memory_improvement_pct", 0)

        print(f"⚡ Performance Change: {time_improvement:+.1f}% execution time")
        print(f"💾 Memory Change: {memory_improvement:+.1f}% peak memory")

    return comparison


# ============================================================================
# PROFILING VISUALIZATION TEMPLATES
# ============================================================================


class ProfilingVisualizationTemplates:
    """Pre-configured visualization templates for common profiling scenarios"""

    @staticmethod
    def training_performance_template(
        profiling_data: ModelProfilingData, mode: str = "exploration"
    ) -> Dict[str, Any]:
        """Template focused on training performance analysis"""

        analyzer = PyTorchProfilingAnalyzer(profiling_data, mode=mode)

        # Training-specific analysis
        analyses = {
            "forward_backward_analysis": analyzer.analyze_layer_performance(),
            "memory_growth_patterns": analyzer.analyze_memory_usage(),
            "gradient_computation_bottlenecks": analyzer.identify_bottlenecks(),
            "training_efficiency_metrics": {},
        }

        # Calculate training-specific metrics
        datasets = analyzer.to_polars_dataframes()
        if "operations" in datasets:
            ops_df = datasets["operations"]

            forward_ops = ops_df.filter(~pl.col("is_backward"))
            backward_ops = ops_df.filter(pl.col("is_backward"))

            analyses["training_efficiency_metrics"] = {
                "forward_time_ms": forward_ops["total_time_ms"].sum(),
                "backward_time_ms": backward_ops["total_time_ms"].sum(),
                "forward_backward_ratio": (
                    forward_ops["total_time_ms"].sum()
                    / (backward_ops["total_time_ms"].sum() + 1e-8)
                ),
                "gradient_ops_count": len(backward_ops),
                "parameter_update_efficiency": (
                    backward_ops["total_time_ms"].mean()
                    / (forward_ops["total_time_ms"].mean() + 1e-8)
                ),
            }

        return analyses

    @staticmethod
    def inference_optimization_template(
        profiling_data: ModelProfilingData, mode: str = "publication"
    ) -> Dict[str, Any]:
        """Template focused on inference optimization"""

        analyzer = PyTorchProfilingAnalyzer(profiling_data, mode=mode)

        # Inference-specific analysis
        analyses = {
            "latency_breakdown": analyzer.analyze_execution_timeline(),
            "memory_efficiency": analyzer.analyze_memory_usage(),
            "throughput_analysis": {},
            "inference_bottlenecks": analyzer.identify_bottlenecks(),
        }

        # Calculate inference-specific metrics
        datasets = analyzer.to_polars_dataframes()
        if "operations" in datasets:
            ops_df = datasets["operations"]

            total_inference_time_ms = ops_df["total_time_ms"].sum()
            batch_size = profiling_data.batch_size or 1

            analyses["throughput_analysis"] = {
                "total_inference_time_ms": total_inference_time_ms,
                "inference_time_per_sample_ms": total_inference_time_ms / batch_size,
                "throughput_samples_per_second": (
                    batch_size / (total_inference_time_ms / 1000.0)
                    if total_inference_time_ms > 0
                    else 0
                ),
                "average_latency_ms": ops_df["total_time_ms"].mean(),
                "latency_p95_ms": ops_df["total_time_ms"].quantile(0.95),
                "latency_p99_ms": ops_df["total_time_ms"].quantile(0.99),
            }

        return analyses

    @staticmethod
    def memory_optimization_template(
        profiling_data: ModelProfilingData, mode: str = "publication"
    ) -> Dict[str, Any]:
        """Template focused on memory optimization opportunities"""

        analyzer = PyTorchProfilingAnalyzer(profiling_data, mode=mode)

        # Memory-focused analysis
        analyses = {
            "memory_usage_patterns": analyzer.analyze_memory_usage(),
            "memory_bottlenecks": analyzer.identify_bottlenecks(),
            "optimization_opportunities": {},
        }

        # Identify memory optimization opportunities
        datasets = analyzer.to_polars_dataframes()
        if "operations" in datasets:
            ops_df = datasets["operations"]
            memory_ops = ops_df.filter(pl.col("gpu_memory_mb") > 0)

            if len(memory_ops) > 0:
                peak_memory = memory_ops["gpu_memory_mb"].max()
                avg_memory = memory_ops["gpu_memory_mb"].mean()

                # Find memory-intensive operation categories
                memory_by_category = (
                    memory_ops.group_by("operation_category")
                    .agg(
                        [
                            pl.col("gpu_memory_mb").max().alias("peak_memory"),
                            pl.col("gpu_memory_mb").mean().alias("avg_memory"),
                            pl.col("total_time_ms").sum().alias("total_time"),
                        ]
                    )
                    .sort("peak_memory", descending=True)
                )

                analyses["optimization_opportunities"] = {
                    "peak_memory_mb": peak_memory,
                    "memory_utilization_ratio": (
                        avg_memory / peak_memory if peak_memory > 0 else 0
                    ),
                    "top_memory_categories": memory_by_category.head(5).to_dict(
                        as_series=False
                    ),
                    "checkpointing_candidates": memory_by_category.filter(
                        pl.col("peak_memory") > peak_memory * 0.2
                    ).to_dict(as_series=False),
                    "memory_optimization_potential_mb": peak_memory - avg_memory,
                }

        return analyses


# ============================================================================
# BATCH ANALYSIS UTILITIES
# ============================================================================


class BatchProfilingAnalyzer:
    """Utilities for analyzing multiple profiling runs"""

    def __init__(self, mode: str = "exploration"):
        self.mode = mode
        self.profiling_runs = []

    def add_profiling_run(
        self,
        trace_file: Union[str, Path],
        run_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add a profiling run to the batch analysis"""

        parser = PyTorchProfilerParser()
        profiling_data = parser.parse_trace_file(trace_file)

        run_info = {
            "name": run_name,
            "data": profiling_data,
            "metadata": metadata or {},
            "trace_file": str(trace_file),
        }

        self.profiling_runs.append(run_info)
        print(f"✅ Added profiling run: {run_name}")

    def analyze_batch_performance(self) -> Dict[str, Any]:
        """Analyze performance across multiple profiling runs"""

        if len(self.profiling_runs) < 2:
            return {"error": "Need at least 2 profiling runs for batch analysis"}

        batch_analysis = {
            "performance_comparison": {},
            "trend_analysis": {},
            "best_configuration": {},
            "optimization_insights": [],
        }

        # Extract metrics from all runs
        run_metrics = []
        for run in self.profiling_runs:
            analyzer = PyTorchProfilingAnalyzer(run["data"], mode=self.mode)
            datasets = analyzer.to_polars_dataframes()

            if "operations" in datasets:
                ops_df = datasets["operations"]

                metrics = {
                    "run_name": run["name"],
                    "total_time_ms": ops_df["total_time_ms"].sum(),
                    "avg_time_ms": ops_df["total_time_ms"].mean(),
                    "peak_memory_mb": ops_df["gpu_memory_mb"].max(),
                    "operation_count": len(ops_df),
                    "metadata": run["metadata"],
                }
                run_metrics.append(metrics)

        # Create comparison DataFrame
        comparison_df = pl.DataFrame(run_metrics)

        # Find best performing configuration
        best_time_run = comparison_df.filter(
            pl.col("total_time_ms") == comparison_df["total_time_ms"].min()
        ).row(0, named=True)

        best_memory_run = comparison_df.filter(
            pl.col("peak_memory_mb") == comparison_df["peak_memory_mb"].min()
        ).row(0, named=True)

        batch_analysis["best_configuration"] = {
            "fastest_execution": best_time_run,
            "most_memory_efficient": best_memory_run,
        }

        # Generate optimization insights
        time_range = (
            comparison_df["total_time_ms"].max() - comparison_df["total_time_ms"].min()
        )
        memory_range = (
            comparison_df["peak_memory_mb"].max()
            - comparison_df["peak_memory_mb"].min()
        )

        insights = []

        if time_range > comparison_df["total_time_ms"].mean() * 0.1:  # >10% variation
            insights.append(
                {
                    "type": "performance_variation",
                    "message": f"Significant performance variation detected: {time_range:.1f}ms range",
                    "recommendation": "Investigate configuration differences affecting execution time",
                }
            )

        if (
            memory_range > comparison_df["peak_memory_mb"].mean() * 0.2
        ):  # >20% variation
            insights.append(
                {
                    "type": "memory_variation",
                    "message": f"High memory usage variation: {memory_range:.1f}MB range",
                    "recommendation": "Consider memory optimization techniques for high-usage configurations",
                }
            )

        batch_analysis["optimization_insights"] = insights
        batch_analysis["performance_comparison"] = comparison_df.to_dict(
            as_series=False
        )

        return batch_analysis

    def generate_batch_report(self, output_dir: Union[str, Path]) -> str:
        """Generate comprehensive batch analysis report"""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        batch_analysis = self.analyze_batch_performance()

        # Create visualizations
        if "performance_comparison" in batch_analysis:
            comparison_data = batch_analysis["performance_comparison"]
            comparison_df = pl.DataFrame(comparison_data)

            # Performance comparison plots
            visualizer = MLVisualizer(self.mode)

            config = MLPlotConfig(
                title="Execution Time Comparison Across Runs",
                xlabel="Run Name",
                ylabel="Total Time (ms)",
                mode=self.mode,
            )

            time_comparison_plot = visualizer.bar_plot(
                comparison_df, config, "run_name", "total_time_ms"
            )

            # Save plot
            plot_path = output_dir / "batch_time_comparison.png"
            visualizer.save_figure(time_comparison_plot, plot_path, config)

            # Memory comparison
            config_memory = MLPlotConfig(
                title="Peak Memory Comparison Across Runs",
                xlabel="Run Name",
                ylabel="Peak Memory (MB)",
                mode=self.mode,
            )

            memory_comparison_plot = visualizer.bar_plot(
                comparison_df, config_memory, "run_name", "peak_memory_mb"
            )

            # Save memory plot
            memory_plot_path = output_dir / "batch_memory_comparison.png"
            visualizer.save_figure(
                memory_comparison_plot, memory_plot_path, config_memory
            )

        # Generate HTML report
        html_report = self._generate_batch_html_report(batch_analysis, output_dir)

        return html_report

    def _generate_batch_html_report(
        self, batch_analysis: Dict[str, Any], output_dir: Path
    ) -> str:
        """Generate HTML report for batch analysis"""

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Batch Profiling Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #e8f4f8; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
                .best-config {{ background-color: #d4edda; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .insight {{ background-color: #fff3cd; padding: 10px; margin: 10px 0; border-left: 4px solid #ffc107; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Batch Profiling Analysis Report</h1>
                <p><strong>Total Runs Analyzed:</strong> {len(self.profiling_runs)}</p>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """

        # Best configurations section
        if "best_configuration" in batch_analysis:
            best_configs = batch_analysis["best_configuration"]
            html_content += """
            <div class="section">
                <h2>Best Performing Configurations</h2>
            """

            if "fastest_execution" in best_configs:
                fastest = best_configs["fastest_execution"]
                html_content += f"""
                <div class="best-config">
                    <h3>🚀 Fastest Execution</h3>
                    <p><strong>Run:</strong> {fastest['run_name']}</p>
                    <p><strong>Total Time:</strong> {fastest['total_time_ms']:.2f} ms</p>
                    <p><strong>Average Operation Time:</strong> {fastest['avg_time_ms']:.3f} ms</p>
                </div>
                """

            if "most_memory_efficient" in best_configs:
                efficient = best_configs["most_memory_efficient"]
                html_content += f"""
                <div class="best-config">
                    <h3>💾 Most Memory Efficient</h3>
                    <p><strong>Run:</strong> {efficient['run_name']}</p>
                    <p><strong>Peak Memory:</strong> {efficient['peak_memory_mb']:.1f} MB</p>
                </div>
                """

            html_content += "</div>"

        # Optimization insights section
        if "optimization_insights" in batch_analysis:
            insights = batch_analysis["optimization_insights"]
            html_content += """
            <div class="section">
                <h2>Optimization Insights</h2>
            """

            if insights:
                for insight in insights:
                    html_content += f"""
                    <div class="insight">
                        <strong>{insight['type'].replace('_', ' ').title()}:</strong><br>
                        {insight['message']}<br>
                        <em>Recommendation: {insight['recommendation']}</em>
                    </div>
                    """
            else:
                html_content += "<p>No significant optimization opportunities identified across runs.</p>"

            html_content += "</div>"

        # Performance comparison table
        if "performance_comparison" in batch_analysis:
            comparison_data = batch_analysis["performance_comparison"]
            html_content += """
            <div class="section">
                <h2>Performance Comparison</h2>
                <table>
                    <tr>
                        <th>Run Name</th>
                        <th>Total Time (ms)</th>
                        <th>Avg Time (ms)</th>
                        <th>Peak Memory (MB)</th>
                        <th>Operations</th>
                    </tr>
            """

            for i in range(len(comparison_data["run_name"])):
                html_content += f"""
                <tr>
                    <td>{comparison_data['run_name'][i]}</td>
                    <td>{comparison_data['total_time_ms'][i]:.2f}</td>
                    <td>{comparison_data['avg_time_ms'][i]:.3f}</td>
                    <td>{comparison_data['peak_memory_mb'][i]:.1f}</td>
                    <td>{comparison_data['operation_count'][i]:,}</td>
                </tr>
                """

            html_content += """
                </table>
            </div>
            """

        html_content += """
        </body>
        </html>
        """

        # Save HTML report
        html_path = output_dir / "batch_analysis_report.html"
        with open(html_path, "w") as f:
            f.write(html_content)

        return str(html_path)


# ============================================================================
# EXAMPLE USAGE AND DOCUMENTATION
# ============================================================================


def demonstrate_profiling_analysis():
    """Demonstrate the profiling analysis system capabilities"""

    print("🚀 PyTorch Profiling Analysis System - Usage Examples")
    print("=" * 65)

    print("\n1️⃣  Basic Profiling Analysis:")
    print(
        """
    # Analyze a single profiler trace
    dashboard = analyze_pytorch_trace(
        trace_file="model_profile.json",
        output_dir="analysis_results",
        mode="exploration"  # or "publication"
    )

    # Access specific analyses
    execution_analysis = dashboard['execution_analysis']
    memory_analysis = dashboard['memory_analysis']
    bottlenecks = dashboard['bottleneck_analysis']
    """
    )

    print("\n2️⃣  Memory Snapshot Analysis:")
    print(
        """
    # Analyze memory snapshots
    memory_results = analyze_memory_snapshot(
        snapshot_file="memory_snapshot.pickle",
        output_dir="memory_analysis",
        mode="publication"
    )
    """
    )

    print("\n3️⃣  Configuration Comparison:")
    print(
        """
    # Compare two different configurations
    comparison = compare_profiling_results(
        baseline_trace="baseline_profile.json",
        optimized_trace="optimized_profile.json",
        output_dir="comparison_results",
        mode="publication"
    )

    # Check improvements
    improvements = comparison['performance_improvements']
    time_improvement = improvements['execution_time_improvement_pct']
    memory_improvement = improvements['memory_improvement_pct']
    """
    )

    print("\n4️⃣  Batch Analysis:")
    print(
        """
    # Analyze multiple profiling runs
    batch_analyzer = BatchProfilingAnalyzer(mode="publication")

    batch_analyzer.add_profiling_run("run1.json", "Baseline")
    batch_analyzer.add_profiling_run("run2.json", "Optimized")
    batch_analyzer.add_profiling_run("run3.json", "Mixed Precision")

    # Generate comprehensive batch report
    report_path = batch_analyzer.generate_batch_report("batch_analysis")
    """
    )

    print("\n5️⃣  Specialized Analysis Templates:")
    print(
        """
    # Training optimization analysis
    training_analysis = ProfilingVisualizationTemplates.training_performance_template(
        profiling_data, mode="exploration"
    )

    # Inference optimization analysis
    inference_analysis = ProfilingVisualizationTemplates.inference_optimization_template(
        profiling_data, mode="publication"
    )

    # Memory optimization analysis
    memory_analysis = ProfilingVisualizationTemplates.memory_optimization_template(
        profiling_data, mode="publication"
    )
    """
    )

    print("\n📊 Available Visualizations:")
    visualizations = [
        "Execution Timeline Analysis - Operation timing patterns",
        "Memory Usage Timeline - GPU/CPU memory over time",
        "Layer Performance Comparison - Per-layer execution analysis",
        "Bottleneck Identification - Automated performance bottleneck detection",
        "Operation Category Distribution - Time spent by operation type",
        "Forward vs Backward Analysis - Training pass comparison",
        "Memory vs Compute Trade-offs - Resource utilization analysis",
        "Configuration Comparisons - A/B testing visualizations",
        "Batch Performance Analysis - Multi-run comparison",
        "Optimization Opportunity Identification - Automated suggestions",
    ]

    for i, viz in enumerate(visualizations, 1):
        print(f"  {i:2d}. {viz}")

    print("\n🎨 Dual-Track Visualization Modes:")
    print("  • Exploration Mode (HoloViews + Datashader):")
    print("    - Interactive plots with zoom, pan, hover")
    print("    - Automatic datashading for large datasets (>50K points)")
    print("    - Fast rendering for iterative analysis")
    print("    - Web-based output (HTML)")

    print("\n  • Publication Mode (Matplotlib + Seaborn):")
    print("    - High-quality static plots for papers/reports")
    print("    - Precise control over styling and typography")
    print("    - Vector output formats (PDF, EPS)")
    print("    - Scientific publication standards")

    print("\n🔧 Key Features:")
    features = [
        "Automatic operation categorization (conv, linear, attention, etc.)",
        "Memory growth pattern analysis and leak detection",
        "Forward/backward pass performance breakdown",
        "Bottleneck identification with optimization suggestions",
        "Large dataset handling with sampling/datashading",
        "Comparative analysis across multiple runs",
        "Comprehensive HTML dashboard generation",
        "Integration with existing ures.plot.backend infrastructure",
    ]

    for feature in features:
        print(f"  ✅ {feature}")


if __name__ == "__main__":
    demonstrate_profiling_analysis()
