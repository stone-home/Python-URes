#!/usr/bin/env python3
"""
conftest.py - Pytest Global Configuration
========================================

Global pytest configuration, fixtures, and hooks for the research plotting system.
This file provides shared fixtures and configuration across all test modules.
"""

import pytest
import warnings
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any, List
import os
import sys

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pandas as pd

# Import system modules to check availability
from ures.plot.backend import (
    HAS_SEABORN,
    HAS_PLOTLY,
    HAS_SCIPY,
    HAS_SKLEARN,
    HAS_STATSMODELS,
    check_dependencies,
)


# ============================================================================
# PYTEST CONFIGURATION HOOKS
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom settings"""
    # Suppress specific warnings during testing
    warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")
    warnings.filterwarnings("ignore", category=FutureWarning, module="plotly")
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Set matplotlib to non-interactive mode
    plt.ioff()

    # Print dependency status at start of test run
    if config.option.verbose >= 1:
        print("\n" + "=" * 60)
        print("🔧 RESEARCH PLOTTING SYSTEM TEST CONFIGURATION")
        print("=" * 60)
        check_dependencies()
        print("=" * 60)


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add automatic markers"""
    for item in items:
        # Add backend-specific markers based on test names
        if "research_backend" in item.name or "research_plotter" in str(item.function):
            item.add_marker(pytest.mark.research_backend)
        if "interactive_backend" in item.name or "interactive_plotter" in str(
            item.function
        ):
            item.add_marker(pytest.mark.interactive_backend)

        # Add slow marker for performance tests
        if "performance" in item.name or "large_dataset" in item.name:
            item.add_marker(pytest.mark.slow)

        # Add visual marker for tests that generate plots
        if any(
            keyword in item.name for keyword in ["plot", "chart", "graph", "visual"]
        ):
            item.add_marker(pytest.mark.visual)


def pytest_runtest_setup(item):
    """Setup hook run before each test"""
    # Skip tests requiring unavailable dependencies
    markers = [mark.name for mark in item.iter_markers()]

    if "requires_seaborn" in markers and not HAS_SEABORN:
        pytest.skip("Test requires seaborn package")
    if "requires_plotly" in markers and not HAS_PLOTLY:
        pytest.skip("Test requires plotly package")
    if "requires_scipy" in markers and not HAS_SCIPY:
        pytest.skip("Test requires scipy package")


def pytest_runtest_teardown(item, nextitem):
    """Cleanup hook run after each test"""
    # Close all matplotlib figures to prevent memory leaks
    plt.close("all")

    # Clear matplotlib cache
    if hasattr(plt, "clf"):
        plt.clf()


def pytest_sessionfinish(session, exitstatus):
    """Hook run after test session completes"""
    print(f"\n📊 Test session completed with exit status: {exitstatus}")

    # Generate summary report
    if hasattr(session.config, "_plot_summary"):
        summary = session.config._plot_summary
        print(f"📈 Total plots generated: {summary.get('total_plots', 0)}")
        print(f"🎯 Test coverage: {summary.get('coverage', 'N/A')}")


# ============================================================================
# COMMAND LINE OPTIONS
# ============================================================================


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--save-plots",
        action="store_true",
        default=False,
        help="Save generated plots for visual inspection",
    )

    parser.addoption(
        "--plot-dir",
        action="store",
        default="./pytest_plots",
        help="Directory to save plots when --save-plots is used",
    )

    parser.addoption(
        "--backend",
        action="store",
        choices=["research", "interactive", "both"],
        default="both",
        help="Which backend(s) to test",
    )

    parser.addoption(
        "--quick",
        action="store_true",
        default=False,
        help="Run only quick tests (skip slow performance tests)",
    )

    parser.addoption(
        "--visual-only",
        action="store_true",
        default=False,
        help="Run only visual tests that generate plots",
    )


# ============================================================================
# SESSION-SCOPED FIXTURES
# ============================================================================


@pytest.fixture(scope="session")
def test_config(request):
    """Session-wide test configuration"""
    return {
        "save_plots": request.config.getoption("--save-plots"),
        "plot_dir": request.config.getoption("--plot-dir"),
        "backend": request.config.getoption("--backend"),
        "quick": request.config.getoption("--quick"),
        "visual_only": request.config.getoption("--visual-only"),
    }


@pytest.fixture(scope="session")
def plot_output_dir(test_config):
    """Create and manage plot output directory"""
    if test_config["save_plots"]:
        plot_dir = Path(test_config["plot_dir"])
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for organization
        (plot_dir / "research").mkdir(exist_ok=True)
        (plot_dir / "interactive").mkdir(exist_ok=True)
        (plot_dir / "comparison").mkdir(exist_ok=True)

        yield plot_dir

        # Session cleanup - optionally remove empty directories
        try:
            if not any(plot_dir.iterdir()):
                plot_dir.rmdir()
        except OSError:
            pass  # Directory not empty, keep it
    else:
        yield None


@pytest.fixture(scope="session")
def temp_session_dir():
    """Session-wide temporary directory"""
    temp_path = Path(tempfile.mkdtemp(prefix="pytest_research_plot_"))
    yield temp_path

    # Cleanup
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="session")
def dependency_status():
    """Session-wide dependency status information"""
    return {
        "seaborn": HAS_SEABORN,
        "plotly": HAS_PLOTLY,
        "scipy": HAS_SCIPY,
        "sklearn": HAS_SKLEARN,
        "statsmodels": HAS_STATSMODELS,
    }


# ============================================================================
# FUNCTION-SCOPED FIXTURES
# ============================================================================


@pytest.fixture(scope="function")
def temp_dir():
    """Function-scoped temporary directory"""
    temp_path = Path(tempfile.mkdtemp(prefix="pytest_plot_test_"))
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="function")
def matplotlib_cleanup():
    """Ensure matplotlib cleanup after each test"""
    yield
    plt.close("all")
    plt.clf()

    # Reset matplotlib rcParams to defaults
    plt.rcdefaults()


# ============================================================================
# SHARED TEST DATA GENERATORS
# ============================================================================


@pytest.fixture(scope="session")
def sample_datasets():
    """Generate all sample datasets once per session"""
    np.random.seed(42)  # Reproducible data

    datasets = {}

    # Time Series Data
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    trend = np.linspace(10, 50, 200)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(200) / 30)
    noise = np.random.normal(0, 2, 200)

    datasets["time_series"] = pl.DataFrame(
        {
            "date": dates,
            "value": trend + seasonal + noise,
            "trend": trend,
            "seasonal": seasonal + 20,
            "noise": noise + 30,
            "category": np.random.choice(["A", "B"], 200),
        }
    )

    # Scientific Experiment Data
    n_samples = 300
    x_exp = np.linspace(0, 10, n_samples)
    y_true = 2 * x_exp + 1
    y_noise = y_true + np.random.normal(0, 1, n_samples)
    errors = np.abs(np.random.normal(0, 0.5, n_samples))

    datasets["experiment"] = pl.DataFrame(
        {
            "x": x_exp,
            "y_true": y_true,
            "y_measured": y_noise,
            "error": errors,
            "residual": y_noise - y_true,
            "group": np.random.choice(["Control", "Treatment"], n_samples),
        }
    )

    # Multivariate Data
    correlation_matrix = np.array(
        [
            [1.0, 0.8, 0.3, 0.1, -0.2],
            [0.8, 1.0, 0.4, 0.2, -0.1],
            [0.3, 0.4, 1.0, 0.6, 0.3],
            [0.1, 0.2, 0.6, 1.0, 0.5],
            [-0.2, -0.1, 0.3, 0.5, 1.0],
        ]
    )

    multi_data = np.random.multivariate_normal([0] * 5, correlation_matrix, 250)
    datasets["multivariate"] = pl.DataFrame(
        {f"var_{i + 1}": multi_data[:, i] for i in range(5)}
    ).with_columns(
        [
            pl.lit(["Type1"] * 83 + ["Type2"] * 83 + ["Type3"] * 84).alias("type"),
            pl.lit(np.random.choice(["GroupA", "GroupB"], 250)).alias("group"),
        ]
    )

    # Distribution Data
    datasets["distributions"] = pl.DataFrame(
        {
            "normal": np.random.normal(0, 1, 1500),
            "exponential": np.random.exponential(2, 1500),
            "uniform": np.random.uniform(-2, 2, 1500),
            "gamma": np.random.gamma(2, 2, 1500),
            "category": np.random.choice(["Alpha", "Beta", "Gamma", "Delta"], 1500),
            "binary": np.random.choice([0, 1], 1500),
        }
    )

    # Large Dataset for performance testing
    datasets["large"] = pl.DataFrame(
        {
            "x": np.random.randn(50000),
            "y": np.random.randn(50000),
            "z": np.random.randn(50000),
            "category": np.random.choice(["A", "B", "C", "D", "E"], 50000),
        }
    )

    return datasets


# Individual dataset fixtures for convenience
@pytest.fixture
def time_series_data(sample_datasets):
    """Time series dataset"""
    return sample_datasets["time_series"]


@pytest.fixture
def experiment_data(sample_datasets):
    """Scientific experiment dataset"""
    return sample_datasets["experiment"]


@pytest.fixture
def multivariate_data(sample_datasets):
    """Multivariate dataset"""
    return sample_datasets["multivariate"]


@pytest.fixture
def distribution_data(sample_datasets):
    """Distribution dataset"""
    return sample_datasets["distributions"]


@pytest.fixture
def large_dataset(sample_datasets):
    """Large dataset for performance testing"""
    return sample_datasets["large"]


# ============================================================================
# PLOTTER FIXTURES WITH BACKEND SELECTION
# ============================================================================


@pytest.fixture(scope="function")
def research_plotter():
    """Research backend plotter"""
    from ures.plot.backend import ResearchPlotter

    plotter = ResearchPlotter(backend="research")
    yield plotter
    # Cleanup any open figures
    if hasattr(plotter.backend, "fig") and plotter.backend.fig:
        plt.close(plotter.backend.fig)


@pytest.fixture(scope="function")
def interactive_plotter(dependency_status):
    """Interactive backend plotter"""
    if not dependency_status["plotly"]:
        pytest.skip("Plotly not available for interactive backend")

    from ures.plot.backend import ResearchPlotter

    return ResearchPlotter(backend="interactive")


@pytest.fixture(scope="function")
def any_plotter(
    request, research_plotter, interactive_plotter, test_config, dependency_status
):
    """Parametrized fixture for testing multiple backends"""
    # Get the backend parameter from parametrization
    if hasattr(request, "param"):
        backend = request.param
        if backend == "research":
            return research_plotter
        elif backend == "interactive":
            if not dependency_status["plotly"]:
                pytest.skip("Plotly not available")
            return interactive_plotter

    # Default to research if no parametrization
    return research_plotter


def pytest_generate_tests(metafunc):
    """Dynamically generate test parameters"""
    if "any_plotter" in metafunc.fixturenames:
        # Determine which backends to test
        backends = ["research"]

        # Add interactive backend if available
        if HAS_PLOTLY:
            backends.append("interactive")

        # Override based on command line option if available
        if hasattr(metafunc.config, "getoption"):
            try:
                backend_choice = metafunc.config.getoption("--backend", default="both")
                if backend_choice == "research":
                    backends = ["research"]
                elif backend_choice == "interactive":
                    backends = ["interactive"] if HAS_PLOTLY else []
            # "both" keeps the full backends list
            except (AttributeError, ValueError):
                # Fallback if option not available
                pass

        # Only parametrize if we have backends to test
        if backends:
            metafunc.parametrize("any_plotter", backends, indirect=True)


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================


@pytest.fixture(scope="function")
def base_plot_config():
    """Standard plot configuration for testing"""
    from ures.plot.backend import PlotConfig, ColorConfig, ColorScheme

    return PlotConfig(
        title="Test Plot",
        xlabel="X Axis",
        ylabel="Y Axis",
        figsize=(10, 6),
        style="whitegrid",
        color_config=ColorConfig(scheme=ColorScheme.SCIENTIFIC),
        grid=True,
        tight_layout=True,
        export_format="png",
        export_dpi=100,
    )


@pytest.fixture(scope="function")
def custom_config_factory():
    """Factory for creating custom configurations"""
    from ures.plot.backend import (
        PlotConfig,
        ColorConfig,
        ColorScheme,
        AnnotationCollection,
    )

    def _create_config(**kwargs):
        defaults = {
            "title": "Custom Test Plot",
            "xlabel": "X",
            "ylabel": "Y",
            "figsize": (8, 6),
            "color_config": ColorConfig(scheme=ColorScheme.SCIENTIFIC),
            "annotations": AnnotationCollection(),
        }
        defaults.update(kwargs)
        return PlotConfig(**defaults)

    return _create_config


# ============================================================================
# UTILITY FIXTURES AND HELPERS
# ============================================================================


@pytest.fixture(scope="function")
def plot_saver():
    """Utility for saving plots during tests"""

    def _save_plot(fig, name: str, backend: str, output_dir: Path = None, config=None):
        if output_dir and output_dir.exists():
            backend_dir = output_dir / backend
            backend_dir.mkdir(exist_ok=True)

            filepath = backend_dir / f"{name}.png"

            try:
                if backend == "research":
                    fig.savefig(filepath, dpi=150, bbox_inches="tight")
                elif backend == "interactive" and HAS_PLOTLY:
                    fig.write_image(str(filepath))
                    # Also save as HTML
                    html_path = backend_dir / f"{name}.html"
                    fig.write_html(str(html_path))

                return filepath
            except Exception as e:
                warnings.warn(f"Failed to save plot {name}: {e}")
                return None
        return None

    return _save_plot


@pytest.fixture(scope="function")
def figure_validator():
    """Utility for validating generated figures"""

    def _validate(fig, backend_name: str):
        assert fig is not None, f"Figure should not be None for {backend_name}"

        if backend_name == "research":
            import matplotlib.figure

            assert isinstance(
                fig, matplotlib.figure.Figure
            ), "Should return matplotlib Figure"
            assert len(fig.get_axes()) > 0, "Figure should have at least one axis"
        elif backend_name == "interactive" and HAS_PLOTLY:
            import plotly.graph_objects as go

            assert isinstance(fig, go.Figure), "Should return plotly Figure"
            assert len(fig.data) > 0, "Figure should have at least one trace"

        return True

    return _validate


# ============================================================================
# SKIP CONDITIONS
# ============================================================================

# Convenience markers for skipping tests
skip_if_no_seaborn = pytest.mark.skipif(not HAS_SEABORN, reason="Seaborn not available")
skip_if_no_plotly = pytest.mark.skipif(not HAS_PLOTLY, reason="Plotly not available")
skip_if_no_scipy = pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
skip_if_no_sklearn = pytest.mark.skipif(
    not HAS_SKLEARN, reason="Scikit-learn not available"
)


# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================


@pytest.fixture(scope="function")
def performance_monitor():
    """Monitor test performance"""
    import time
    import psutil
    import os

    process = psutil.Process(os.getpid())

    start_time = time.time()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB

    yield

    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB

    duration = end_time - start_time
    memory_diff = end_memory - start_memory

    # Log performance metrics for slow tests
    if duration > 5.0 or memory_diff > 100:  # 5 seconds or 100MB
        print(f"\n⚠️ Performance Alert:")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Memory change: {memory_diff:.1f}MB")


# ============================================================================
# ERROR COLLECTION AND REPORTING
# ============================================================================


@pytest.fixture(scope="session", autouse=True)
def error_collector():
    """Collect and report errors across the test session"""
    errors = []

    yield errors

    # Report collected errors at the end
    if errors:
        print(f"\n❌ COLLECTED ERRORS ({len(errors)}):")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
