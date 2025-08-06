#!/usr/bin/env python3
"""
Research Plotting System - Fixed Pytest Test Suite
=================================================

Fixed version that resolves duplicate parametrization issues.
All parametrization logic is handled in conftest.py to avoid conflicts.

Usage:
    pytest test_utils_fixed.py -v
    pytest test_utils_fixed.py --save-plots --plot-dir ./test_outputs -v
"""

import pytest
from pathlib import Path
from typing import Optional
import warnings

import polars as pl
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

# Import our research plotting system
from ures.plot.backend import (
    ResearchPlotter,
    PlotConfig,
    ColorConfig,
    ColorScheme,
    AnnotationCollection,
    create_quick_config,
    HAS_SEABORN,
    HAS_PLOTLY,
    HAS_SCIPY,
    HAS_SKLEARN,
    HAS_STATSMODELS,
)


# ============================================================================
# UTILITY FUNCTIONS FOR TESTING
# ============================================================================


def save_plot_if_requested(
    fig,
    plot_name: str,
    backend_name: str,
    plot_output_dir: Optional[Path],
    config: PlotConfig,
):
    """Save plot if --save-plots option is specified"""
    if plot_output_dir:
        backend_dir = plot_output_dir / backend_name
        backend_dir.mkdir(exist_ok=True)

        filename = f"{plot_name}.png"
        filepath = backend_dir / filename

        try:
            if backend_name == "research":
                fig.savefig(filepath, dpi=config.export_dpi, bbox_inches="tight")
            elif backend_name == "interactive" and HAS_PLOTLY:
                fig.write_html(backend_dir / f"{plot_name}.html")
                fig.write_image(str(filepath))  # Requires kaleido
        except Exception as e:
            warnings.warn(f"Failed to save plot {filename}: {e}")


def validate_figure(fig, backend_name: str):
    """Validate that figure was created successfully"""
    assert fig is not None, f"Figure should not be None for {backend_name} backend"

    if backend_name == "research":
        # Matplotlib figure validation
        import matplotlib.figure

        assert isinstance(
            fig, matplotlib.figure.Figure
        ), "Should return matplotlib Figure"
        assert len(fig.get_axes()) > 0, "Figure should have at least one axis"

    elif backend_name == "interactive":
        # Plotly figure validation
        if HAS_PLOTLY:
            import plotly.graph_objects as go

            assert isinstance(fig, go.Figure), "Should return plotly Figure"
            assert len(fig.data) > 0, "Figure should have at least one trace"


# ============================================================================
# SYSTEM TESTS
# ============================================================================


class TestSystemSetup:
    """Test system setup and dependencies"""

    def test_imports(self):
        """Test that all required modules can be imported"""
        from ures.plot.backend import (
            ResearchPlotter,
            PlotConfig,
            ColorConfig,
            ColorScheme,
            AnnotationCollection,
        )

        assert True  # If we get here, imports worked

    def test_dependency_flags(self):
        """Test dependency detection flags"""
        # These should always be boolean
        assert isinstance(HAS_SEABORN, bool)
        assert isinstance(HAS_PLOTLY, bool)
        assert isinstance(HAS_SCIPY, bool)
        assert isinstance(HAS_SKLEARN, bool)
        assert isinstance(HAS_STATSMODELS, bool)

    def test_research_backend_creation(self):
        """Test research backend can be created"""
        plotter = ResearchPlotter(backend="research")
        assert plotter is not None
        assert plotter.backend_name == "research"

    @pytest.mark.skipif(not HAS_PLOTLY, reason="Plotly not available")
    def test_interactive_backend_creation(self):
        """Test interactive backend can be created"""
        plotter = ResearchPlotter(backend="interactive")
        assert plotter is not None
        assert plotter.backend_name == "interactive"

    def test_backend_info(self):
        """Test backend information retrieval"""
        plotter = ResearchPlotter(backend="research")
        info = plotter.get_backend_info()
        assert isinstance(info, dict)
        assert "name" in info
        assert "type" in info
        assert info["name"] == "research"


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================


class TestConfiguration:
    """Test configuration classes and functionality"""

    def test_plot_config_creation(self):
        """Test PlotConfig can be created with defaults"""
        config = PlotConfig(
            title="Test Plot", xlabel="X Axis", ylabel="Y Axis", figsize=(8, 6)
        )
        assert config.title == "Test Plot"
        assert config.figsize == (8, 6)
        assert config.grid is True

    def test_color_config(self):
        """Test color configuration"""
        color_config = ColorConfig(scheme=ColorScheme.SCIENTIFIC)
        colors = color_config.get_colors()
        assert isinstance(colors, list)
        assert len(colors) > 0
        assert all(isinstance(c, str) for c in colors)

        # Test color cycling
        color1 = color_config.get_color(0)
        color2 = color_config.get_color(len(colors))  # Should cycle back
        assert color1 == color2

    def test_annotation_collection(self):
        """Test annotation system"""
        annotations = AnnotationCollection()

        # Test adding different annotation types
        annotations.add_text("Test", 0.5, 0.5)
        annotations.add_arrow("Arrow", 0.3, 0.7, (0.4, 0.8))
        annotations.add_box("Box", 0.6, 0.4)

        assert len(annotations.annotations) == 3
        assert annotations.annotations[0].annotation_type == "text"
        assert annotations.annotations[1].annotation_type == "arrow"
        assert annotations.annotations[2].annotation_type == "box"

        # Test clearing
        annotations.clear()
        assert len(annotations.annotations) == 0

    def test_create_quick_config(self):
        """Test quick config creation utility"""
        config = create_quick_config("Quick Test", "X", "Y", figsize=(12, 8))
        assert config.title == "Quick Test"
        assert config.xlabel == "X"
        assert config.ylabel == "Y"
        assert config.figsize == (12, 8)


# ============================================================================
# BASIC PLOT TESTS
# ============================================================================


class TestBasicPlots:
    """Test basic plotting functionality"""

    def test_line_plot_creation(
        self, any_plotter, time_series_data, base_plot_config, plot_output_dir
    ):
        """Test line plot creation - uses any_plotter from conftest.py"""
        fig = any_plotter.line_plot(time_series_data, base_plot_config, "date", "value")

        validate_figure(fig, any_plotter.backend_name)
        save_plot_if_requested(
            fig,
            "line_plot",
            any_plotter.backend_name,
            plot_output_dir,
            base_plot_config,
        )

        # Clean up matplotlib figures
        if any_plotter.backend_name == "research":
            plt.close(fig)

    def test_scatter_plot_creation(
        self, any_plotter, experiment_data, base_plot_config, plot_output_dir
    ):
        """Test scatter plot creation"""
        fig = any_plotter.scatter_plot(
            experiment_data, base_plot_config, "x", "y_measured"
        )

        validate_figure(fig, any_plotter.backend_name)
        save_plot_if_requested(
            fig,
            "scatter_plot",
            any_plotter.backend_name,
            plot_output_dir,
            base_plot_config,
        )

        if any_plotter.backend_name == "research":
            plt.close(fig)

    def test_bar_plot_creation(
        self, any_plotter, distribution_data, base_plot_config, plot_output_dir
    ):
        """Test bar plot creation"""
        # Create aggregated data for bar plot
        bar_data = distribution_data.group_by("category").agg(
            pl.col("normal").mean().alias("mean_value")
        )

        fig = any_plotter.bar_plot(bar_data, base_plot_config, "category", "mean_value")

        validate_figure(fig, any_plotter.backend_name)
        save_plot_if_requested(
            fig, "bar_plot", any_plotter.backend_name, plot_output_dir, base_plot_config
        )

        if any_plotter.backend_name == "research":
            plt.close(fig)

    def test_histogram_creation(
        self, any_plotter, distribution_data, base_plot_config, plot_output_dir
    ):
        """Test histogram creation"""
        fig = any_plotter.histogram(distribution_data, base_plot_config, "normal")

        validate_figure(fig, any_plotter.backend_name)
        save_plot_if_requested(
            fig,
            "histogram",
            any_plotter.backend_name,
            plot_output_dir,
            base_plot_config,
        )

        if any_plotter.backend_name == "research":
            plt.close(fig)


# ============================================================================
# DISTRIBUTION PLOT TESTS
# ============================================================================


class TestDistributionPlots:
    """Test distribution visualization plots"""

    def test_box_plot_creation(
        self, any_plotter, distribution_data, base_plot_config, plot_output_dir
    ):
        """Test box plot creation"""
        fig = any_plotter.box_plot(
            distribution_data, base_plot_config, "normal", "category"
        )

        validate_figure(fig, any_plotter.backend_name)
        save_plot_if_requested(
            fig, "box_plot", any_plotter.backend_name, plot_output_dir, base_plot_config
        )

        if any_plotter.backend_name == "research":
            plt.close(fig)

    def test_violin_plot_creation(
        self, any_plotter, distribution_data, base_plot_config, plot_output_dir
    ):
        """Test violin plot creation"""
        fig = any_plotter.violin_plot(
            distribution_data, base_plot_config, "normal", "category"
        )

        validate_figure(fig, any_plotter.backend_name)
        save_plot_if_requested(
            fig,
            "violin_plot",
            any_plotter.backend_name,
            plot_output_dir,
            base_plot_config,
        )

        if any_plotter.backend_name == "research":
            plt.close(fig)


# ============================================================================
# CORRELATION AND RELATIONSHIP TESTS
# ============================================================================


class TestCorrelationPlots:
    """Test correlation and relationship plots"""

    def test_heatmap_creation(
        self, any_plotter, multivariate_data, base_plot_config, plot_output_dir
    ):
        """Test heatmap creation"""
        # Use only numeric columns
        numeric_data = multivariate_data.select(
            ["var_1", "var_2", "var_3", "var_4", "var_5"]
        )

        fig = any_plotter.heatmap(numeric_data, base_plot_config, correlation=True)

        validate_figure(fig, any_plotter.backend_name)
        save_plot_if_requested(
            fig, "heatmap", any_plotter.backend_name, plot_output_dir, base_plot_config
        )

        if any_plotter.backend_name == "research":
            plt.close(fig)

    def test_regression_plot_creation(
        self, any_plotter, experiment_data, base_plot_config, plot_output_dir
    ):
        """Test regression plot creation"""
        fig = any_plotter.regression_plot(
            experiment_data, base_plot_config, "x", "y_measured"
        )

        validate_figure(fig, any_plotter.backend_name)
        save_plot_if_requested(
            fig,
            "regression_plot",
            any_plotter.backend_name,
            plot_output_dir,
            base_plot_config,
        )

        if any_plotter.backend_name == "research":
            plt.close(fig)

    @pytest.mark.skipif(not HAS_SEABORN, reason="Seaborn required for pair plots")
    def test_pair_plot_creation(
        self, multivariate_data, base_plot_config, plot_output_dir
    ):
        """Test pair plot creation (research backend only)"""
        # Only test with research backend since pair plots are seaborn-specific
        plotter = ResearchPlotter(backend="research")

        fig = plotter.pair_plot(
            multivariate_data, base_plot_config, ["var_1", "var_2", "var_3"]
        )

        validate_figure(fig, plotter.backend_name)
        save_plot_if_requested(
            fig, "pair_plot", plotter.backend_name, plot_output_dir, base_plot_config
        )

        plt.close(fig)


# ============================================================================
# STYLING AND CUSTOMIZATION TESTS
# ============================================================================


class TestStylingAndCustomization:
    """Test styling and customization features"""

    @pytest.mark.parametrize(
        "color_scheme",
        [
            ColorScheme.SCIENTIFIC,
            ColorScheme.NATURE,
            ColorScheme.COLORBLIND_FRIENDLY,
            ColorScheme.VIRIDIS,
        ],
    )
    def test_color_schemes(self, experiment_data, color_scheme, plot_output_dir):
        """Test different color schemes"""
        plotter = ResearchPlotter(backend="research")
        config = PlotConfig(
            title=f"Color Test - {color_scheme.value}",
            color_config=ColorConfig(scheme=color_scheme),
        )

        fig = plotter.scatter_plot(experiment_data, config, "x", "y_measured")

        validate_figure(fig, plotter.backend_name)
        save_plot_if_requested(
            fig,
            f"color_{color_scheme.value}",
            plotter.backend_name,
            plot_output_dir,
            config,
        )

        plt.close(fig)

    def test_annotations(self, experiment_data, plot_output_dir):
        """Test annotation functionality"""
        plotter = ResearchPlotter(backend="research")
        config = PlotConfig(title="Annotation Test")

        # Add various annotations
        config.annotations.add_text("Peak", 5, 12, fontsize=12, color="red")
        config.annotations.add_arrow("Trend", 8, 16, (2, 5), color="blue")
        config.annotations.add_box(
            "Region",
            7,
            14,
            bbox_props={
                "boxstyle": "round,pad=0.3",
                "facecolor": "yellow",
                "alpha": 0.5,
            },
        )

        fig = plotter.scatter_plot(experiment_data, config, "x", "y_measured")

        validate_figure(fig, plotter.backend_name)
        save_plot_if_requested(
            fig, "annotations", plotter.backend_name, plot_output_dir, config
        )

        plt.close(fig)

    def test_units_and_labels(self, time_series_data, plot_output_dir):
        """Test unit and label functionality"""
        plotter = ResearchPlotter(backend="research")
        config = PlotConfig(
            title="Units Test",
            xlabel="Time",
            ylabel="Temperature",
            x_unit="days",
            y_unit="°C",
        )

        fig = plotter.line_plot(time_series_data, config, "date", "value")

        validate_figure(fig, plotter.backend_name)
        save_plot_if_requested(
            fig, "units", plotter.backend_name, plot_output_dir, config
        )

        plt.close(fig)


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_invalid_column_names(self, experiment_data):
        """Test handling of invalid column names"""
        plotter = ResearchPlotter(backend="research")
        config = PlotConfig(title="Error Test")

        with pytest.raises((KeyError, ValueError, Exception)):
            plotter.scatter_plot(experiment_data, config, "nonexistent_x", "y_measured")

    def test_empty_dataframe(self):
        """
        Fixed version of empty DataFrame test

        The issue was that matplotlib can sometimes handle empty data gracefully
        without raising an exception. We need to add explicit validation.
        """
        import polars as pl
        from ures.plot.backend import ResearchPlotter, PlotConfig

        plotter = ResearchPlotter(backend="research")
        config = PlotConfig(title="Empty Test")
        empty_df = pl.DataFrame({"x": [], "y": []})

        # This should now raise ValueError due to validation
        try:
            fig = plotter.scatter_plot(empty_df, config, "x", "y")
            assert False, "Should have raised ValueError for empty DataFrame"
        except ValueError as e:
            assert "empty DataFrame" in str(e)
            print("✅ Empty DataFrame test now correctly raises ValueError")

    def test_backend_switching(self):
        """Test backend switching functionality"""
        plotter = ResearchPlotter("research")
        assert plotter.backend_name == "research"

        if HAS_PLOTLY:
            plotter.switch_backend("interactive")
            assert plotter.backend_name == "interactive"

    def test_invalid_backend(self):
        """Test invalid backend handling"""
        with pytest.raises(ValueError):
            ResearchPlotter("nonexistent_backend")


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


class TestPerformance:
    """Test performance characteristics"""

    def test_large_dataset_handling(self, large_dataset):
        """Test handling of large datasets"""
        plotter = ResearchPlotter(backend="research")
        config = PlotConfig(title="Large Dataset Test")

        # This should complete without memory issues
        fig = plotter.scatter_plot(large_dataset, config, "x", "y")
        validate_figure(fig, plotter.backend_name)
        plt.close(fig)

    @pytest.mark.timeout(30)  # 30 second timeout
    def test_plot_creation_performance(self, experiment_data):
        """Test that plot creation completes in reasonable time"""
        plotter = ResearchPlotter(backend="research")
        config = PlotConfig(title="Performance Test")

        fig = plotter.scatter_plot(experiment_data, config, "x", "y_measured")
        validate_figure(fig, plotter.backend_name)
        plt.close(fig)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Test integration between components"""

    def test_complete_workflow(self, time_series_data, temp_dir):
        """Test complete workflow from data to exported plot"""
        plotter = ResearchPlotter(backend="research")

        # Create config with all features
        config = PlotConfig(
            title="Complete Workflow Test",
            xlabel="Time",
            ylabel="Value",
            color_config=ColorConfig(scheme=ColorScheme.NATURE),
            figsize=(12, 8),
            export_format="png",
            export_dpi=150,
        )

        # Add annotations
        config.annotations.add_text("Important Point", 50, 30)

        # Create plot
        fig = plotter.line_plot(time_series_data, config, "date", "value")

        # Export plot
        export_path = temp_dir / "workflow_test.png"
        plotter.export(fig, str(export_path), config)

        # Verify file was created
        assert export_path.exists()
        assert export_path.stat().st_size > 0


# ============================================================================
# RUN TESTS WHEN EXECUTED DIRECTLY
# ============================================================================

if __name__ == "__main__":
    # Run tests when executed directly
    import subprocess
    import sys

    # Run with verbose output and save plots
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        __file__,
        "-v",
        "--save-plots",
        "--plot-dir=./pytest_visual_output",
    ]

    print("Running pytest with plot generation...")
    subprocess.run(cmd)

    print("\n" + "=" * 60)
    print("📊 Pytest execution complete!")
    print("📁 Visual plots saved to: ./pytest_visual_output/")
    print("🔍 Review the generated plots for visual verification.")
