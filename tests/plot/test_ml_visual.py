"""
ML Visualization System - Pytest Test Suite
Tests both exploration and publication modes with various chart types
Run with: pytest test_ml_visual.py -v
"""

import pytest
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path


# Test fixtures and data
@pytest.fixture(scope="session")
def output_dir():
    """Create output directory for test files"""
    test_output = Path("test_outputs")
    test_output.mkdir(exist_ok=True)
    return test_output


@pytest.fixture(scope="session")
def sample_data():
    """Generate synthetic test data for various ML scenarios"""
    np.random.seed(42)

    # Training curves data
    epochs = np.arange(1, 51)  # Smaller dataset for faster tests
    train_loss = (
        2.0 * np.exp(-epochs / 15) + 0.1 + np.random.normal(0, 0.05, len(epochs))
    )
    val_loss = (
        2.2 * np.exp(-epochs / 12) + 0.15 + np.random.normal(0, 0.08, len(epochs))
    )
    train_acc = 1 - np.exp(-epochs / 10) * 0.8 + np.random.normal(0, 0.02, len(epochs))
    val_acc = 1 - np.exp(-epochs / 9) * 0.85 + np.random.normal(0, 0.03, len(epochs))

    training_df = pl.DataFrame(
        {
            "epoch": epochs,
            "train_loss": np.maximum(train_loss, 0.05),
            "val_loss": np.maximum(val_loss, 0.05),
            "train_accuracy": np.clip(train_acc, 0, 1),
            "val_accuracy": np.clip(val_acc, 0, 1),
        }
    )

    # Model comparison data
    models = ["SVM", "Random Forest", "XGBoost"]
    accuracy = [0.82, 0.89, 0.94]
    std_err = [0.03, 0.02, 0.02]

    model_comparison_df = pl.DataFrame(
        {"model": models, "accuracy": accuracy, "std_error": std_err}
    )

    # Hyperparameter optimization data
    learning_rates = [0.01, 0.1]
    batch_sizes = [32, 64]

    hyperparam_data = []
    for lr in learning_rates:
        for bs in batch_sizes:
            performance = 0.8 + 0.1 * np.random.random()
            hyperparam_data.append(
                {
                    "learning_rate": lr,
                    "batch_size": bs,
                    "accuracy": min(max(performance, 0.7), 0.95),
                }
            )

    hyperparam_df = pl.DataFrame(hyperparam_data)

    # Statistical test data
    n_points = 100  # Smaller for faster tests
    statistical_df = pl.DataFrame(
        {
            "feature1": np.random.normal(5, 2, n_points),
            "feature2": np.random.exponential(2, n_points),
            "feature3": np.random.uniform(0, 10, n_points),
            "target": np.random.normal(10, 3, n_points),
            "group": np.random.choice(["Group1", "Group2"], n_points),
        }
    )

    # Time series data (simplified)
    n_days = 50
    time_values = np.arange(n_days)
    trend = np.linspace(100, 120, n_days)
    seasonal = 5 * np.sin(2 * np.pi * time_values / 10)
    noise = np.random.normal(0, 2, n_days)

    time_series_df = pl.DataFrame(
        {
            "time": time_values,
            "value": trend + seasonal + noise,
            "trend_component": trend,
            "seasonal_component": seasonal,
        }
    )

    return {
        "training": training_df,
        "model_comparison": model_comparison_df,
        "hyperparam": hyperparam_df,
        "statistical": statistical_df,
        "time_series": time_series_df,
    }


@pytest.fixture
def ml_config():
    """Basic ML plot configuration"""
    from ures.plot.ml_visual import MLPlotConfig

    return MLPlotConfig(
        title="Test Plot", xlabel="X Axis", ylabel="Y Axis", figsize=(8, 6)
    )


# Import tests
class TestImports:
    """Test that all required modules can be imported"""

    def test_import_ml_visual(self):
        """Test ml_visual module imports"""
        try:
            from ml_visual import MLVisualizer, MLPlotConfig, PaperStyleConfig

            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import ml_visual: {e}")

    def test_import_utils(self):
        """Test utils module imports"""
        try:
            from utils import ResearchPlotter, PlotConfig, ColorConfig

            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import utils: {e}")


# Configuration tests
class TestConfiguration:
    """Test configuration system"""

    def test_ml_plot_config_creation(self):
        """Test MLPlotConfig creation"""
        from ures.plot.ml_visual import MLPlotConfig

        config = MLPlotConfig(
            title="Test", xlabel="X", ylabel="Y", experiment_name="test_exp"
        )

        assert config.title == "Test"
        assert config.xlabel == "X"
        assert config.ylabel == "Y"
        assert config.experiment_name == "test_exp"
        assert config.mode == "exploration"  # default

    def test_publication_config_generation(self):
        """Test publication config generation"""
        from ures.plot.ml_visual import MLPlotConfig

        config = MLPlotConfig(title="Test")
        pub_config = config.for_publication()

        assert pub_config.mode == "publication"
        assert pub_config.export_format == "pdf"
        assert pub_config.export_dpi == 300

    def test_paper_style_config(self):
        """Test paper style configuration"""
        from ures.plot.ml_visual import PaperStyleConfig

        paper_style = PaperStyleConfig()

        assert paper_style.single_column_width == 3.5
        assert paper_style.double_column_width == 7.0
        assert paper_style.font_family == "Arial"
        assert len(paper_style.colors) >= 5


# Backend tests
class TestBackends:
    """Test different plotting backends"""

    def test_ml_visualizer_initialization(self):
        """Test MLVisualizer initialization"""
        from ures.plot.ml_visual import MLVisualizer

        # Test exploration mode
        viz_exp = MLVisualizer(mode="exploration")
        assert viz_exp.mode == "exploration"

        # Test publication mode
        viz_pub = MLVisualizer(mode="publication")
        assert viz_pub.mode == "publication"

    def test_mode_switching(self):
        """Test switching between modes"""
        from ures.plot.ml_visual import MLVisualizer

        viz = MLVisualizer(mode="exploration")
        assert viz.mode == "exploration"

        viz.switch_mode("publication")
        assert viz.mode == "publication"

        viz.switch_mode("exploration")
        assert viz.mode == "exploration"

    @pytest.mark.skipif(
        not pytest.importorskip("ml_visual", reason="ml_visual not available"),
        reason="Skip if HoloViews not available",
    )
    def test_exploration_backend_creation(self):
        """Test exploration backend creation"""
        try:
            from ml_visual import ExplorationBackend

            backend = ExplorationBackend()
            assert backend is not None
        except ImportError:
            pytest.skip("HoloViews not available")

    def test_publication_backend_creation(self):
        """Test publication backend creation"""
        from ures.plot.ml_visual import PublicationBackend

        backend = PublicationBackend()
        assert backend is not None


# Basic plotting tests
class TestBasicPlots:
    """Test basic plotting functionality"""

    def test_scatter_plot(self, sample_data, ml_config, output_dir):
        """Test scatter plot creation"""
        from ures.plot.ml_visual import MLVisualizer

        viz = MLVisualizer(mode="publication")
        df = sample_data["statistical"]

        fig = viz.scatter_plot(df, ml_config, "feature1", "feature2")

        assert fig is not None
        if hasattr(fig, "savefig"):
            fig.savefig(output_dir / "test_scatter.png")
            plt.close(fig)

    def test_line_plot(self, sample_data, ml_config, output_dir):
        """Test line plot creation"""
        from ures.plot.ml_visual import MLVisualizer

        viz = MLVisualizer(mode="publication")
        df = sample_data["time_series"]

        fig = viz.line_plot(df, ml_config, "time", "value")

        assert fig is not None
        if hasattr(fig, "savefig"):
            fig.savefig(output_dir / "test_line.png")
            plt.close(fig)

    def test_histogram(self, sample_data, ml_config, output_dir):
        """Test histogram creation"""
        from ures.plot.ml_visual import MLVisualizer

        viz = MLVisualizer(mode="publication")
        df = sample_data["statistical"]

        fig = viz.histogram(df, ml_config, "feature1")

        assert fig is not None
        if hasattr(fig, "savefig"):
            fig.savefig(output_dir / "test_histogram.png")
            plt.close(fig)

    def test_box_plot(self, sample_data, ml_config, output_dir):
        """Test box plot creation"""
        from ures.plot.ml_visual import MLVisualizer

        viz = MLVisualizer(mode="publication")
        df = sample_data["statistical"]

        fig = viz.box_plot(df, ml_config, "feature2", "group")

        assert fig is not None
        if hasattr(fig, "savefig"):
            fig.savefig(output_dir / "test_box.png")
            plt.close(fig)


# ML-specific plotting tests
class TestMLPlots:
    """Test ML-specific plotting functionality"""

    def test_training_curves(self, sample_data, ml_config, output_dir):
        """Test training curves plot"""
        from ures.plot.ml_visual import MLVisualizer

        viz = MLVisualizer(mode="publication")
        df = sample_data["training"]

        fig = viz.training_curves(
            df, ml_config, x_col="epoch", y_cols=["train_loss", "val_loss"]
        )

        assert fig is not None
        if hasattr(fig, "savefig"):
            fig.savefig(output_dir / "test_training_curves.png")
            plt.close(fig)

    def test_model_comparison(self, sample_data, ml_config, output_dir):
        """Test model comparison plot"""
        from ures.plot.ml_visual import MLVisualizer

        viz = MLVisualizer(mode="publication")
        df = sample_data["model_comparison"]

        fig = viz.model_comparison(df, ml_config, "model", "accuracy", "std_error")

        assert fig is not None
        if hasattr(fig, "savefig"):
            fig.savefig(output_dir / "test_model_comparison.png")
            plt.close(fig)

    def test_hyperparameter_heatmap(self, sample_data, ml_config, output_dir):
        """Test hyperparameter heatmap"""
        from ures.plot.ml_visual import MLVisualizer

        viz = MLVisualizer(mode="publication")
        df = sample_data["hyperparam"]

        fig = viz.hyperparameter_heatmap(
            df, ml_config, "batch_size", "learning_rate", "accuracy"
        )

        assert fig is not None
        if hasattr(fig, "savefig"):
            fig.savefig(output_dir / "test_hyperparam_heatmap.png")
            plt.close(fig)


# Advanced plotting tests
class TestAdvancedPlots:
    """Test advanced plotting functionality"""

    def test_bubble_plot(self, sample_data, ml_config, output_dir):
        """Test bubble plot creation"""
        from ures.plot.ml_visual import MLVisualizer

        viz = MLVisualizer(mode="publication")
        df = sample_data["statistical"]

        # Add size column for bubble plot
        df_with_size = df.with_columns([(pl.col("feature3") * 10).alias("bubble_size")])

        fig = viz.bubble_plot(
            df_with_size, ml_config, "feature1", "feature2", "bubble_size"
        )

        assert fig is not None
        if hasattr(fig, "savefig"):
            fig.savefig(output_dir / "test_bubble.png")
            plt.close(fig)

    def test_correlation_heatmap(self, sample_data, ml_config, output_dir):
        """Test correlation heatmap"""
        from ures.plot.ml_visual import MLVisualizer

        viz = MLVisualizer(mode="publication")
        df = sample_data["statistical"]

        # Select only numeric columns
        numeric_df = df.select(["feature1", "feature2", "feature3", "target"])

        fig = viz.heatmap(numeric_df, ml_config, correlation=True)

        assert fig is not None
        if hasattr(fig, "savefig"):
            fig.savefig(output_dir / "test_correlation_heatmap.png")
            plt.close(fig)

    @pytest.mark.skipif(
        not pytest.importorskip("scipy", reason="scipy not available"),
        reason="Skip if scipy not available",
    )
    def test_regression_plot(self, sample_data, ml_config, output_dir):
        """Test regression plot"""
        from ures.plot.ml_visual import MLVisualizer

        viz = MLVisualizer(mode="publication")
        df = sample_data["statistical"]

        fig = viz.regression_plot(df, ml_config, "feature1", "target")

        assert fig is not None
        if hasattr(fig, "savefig"):
            fig.savefig(output_dir / "test_regression.png")
            plt.close(fig)


# Subplot tests
class TestSubplots:
    """Test subplot functionality"""

    def test_subplots_grid(self, sample_data, ml_config, output_dir):
        """Test subplot grid creation"""
        from ures.plot.ml_visual import MLVisualizer

        viz = MLVisualizer(mode="publication")
        df = sample_data["statistical"]

        plot_configs = [
            {
                "type": "scatter",
                "params": {"x_col": "feature1", "y_col": "feature2"},
                "title": "Scatter",
            },
            {
                "type": "hist",
                "params": {"col": "feature1", "bins": 20},
                "title": "Histogram",
            },
            {
                "type": "line",
                "params": {"x_col": "feature1", "y_col": "target"},
                "title": "Line",
            },
            {"type": "box", "params": {"col": "feature2"}, "title": "Box Plot"},
        ]

        fig = viz.subplots_grid(df, ml_config, plot_configs, nrows=2, ncols=2)

        assert fig is not None
        if hasattr(fig, "savefig"):
            fig.savefig(output_dir / "test_subplots_grid.png")
            plt.close(fig)

    def test_pair_plot(self, sample_data, ml_config, output_dir):
        """Test pair plot creation"""
        from ures.plot.ml_visual import MLVisualizer

        viz = MLVisualizer(mode="publication")
        df = sample_data["statistical"]

        # Use fewer columns for faster testing
        columns = ["feature1", "feature2", "feature3"]

        fig = viz.pair_plot(df, ml_config, columns)

        assert fig is not None
        if hasattr(fig, "savefig"):
            fig.savefig(output_dir / "test_pair_plot.png")
            plt.close(fig)


# Error handling and edge cases
class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_empty_dataframe(self):
        """Test handling of empty dataframes"""
        from ures.plot.ml_visual import MLVisualizer, MLPlotConfig

        viz = MLVisualizer(mode="publication")
        config = MLPlotConfig(title="Empty Test")

        # Empty dataframe
        empty_df = pl.DataFrame({"x": [], "y": []})

        with pytest.warns(UserWarning):
            # Should handle gracefully or warn
            try:
                fig = viz.scatter_plot(empty_df, config, "x", "y")
                if fig and hasattr(fig, "savefig"):
                    plt.close(fig)
            except Exception:
                pass  # Expected to fail gracefully

    def test_missing_columns(self, sample_data):
        """Test handling of missing columns"""
        from ures.plot.ml_visual import MLVisualizer, MLPlotConfig

        viz = MLVisualizer(mode="publication")
        config = MLPlotConfig(title="Missing Column Test")
        df = sample_data["statistical"]

        with pytest.raises(Exception):
            # Should raise error for non-existent column
            viz.scatter_plot(df, config, "nonexistent_col", "feature1")

    def test_invalid_mode(self):
        """Test invalid mode handling"""
        from ures.plot.ml_visual import MLVisualizer

        viz = MLVisualizer(mode="publication")

        with pytest.raises(Exception):
            # Should handle invalid mode gracefully
            viz.switch_mode("invalid_mode")


# Integration tests
class TestIntegration:
    """Test integration between components"""

    def test_exploration_to_publication_workflow(self, sample_data, output_dir):
        """Test complete workflow from exploration to publication"""
        from ures.plot.ml_visual import MLVisualizer, MLPlotConfig

        viz = MLVisualizer(mode="exploration")
        config = MLPlotConfig(
            title="Integration Test", xlabel="Feature 1", ylabel="Feature 2"
        )

        df = sample_data["statistical"]

        # Use exploration_to_publication method
        pub_fig = viz.exploration_to_publication(
            viz.scatter_plot, df, config, "feature1", "feature2"
        )

        assert pub_fig is not None
        if hasattr(pub_fig, "savefig"):
            pub_fig.savefig(output_dir / "test_integration.png")
            plt.close(pub_fig)

    def test_publication_save_workflow(self, sample_data, output_dir):
        """Test publication saving workflow"""
        from ures.plot.ml_visual import MLVisualizer, MLPlotConfig

        viz = MLVisualizer(mode="publication")
        config = MLPlotConfig(
            title="Publication Save Test", xlabel="Epoch", ylabel="Accuracy"
        )

        df = sample_data["training"]

        fig = viz.training_curves(
            df, config, x_col="epoch", y_cols=["train_accuracy", "val_accuracy"]
        )

        # Test publication save
        pub_config = config.for_publication()
        viz.save_publication_figure(
            fig, output_dir / "test_publication_save", pub_config
        )

        # Check if file was created
        assert (output_dir / "test_publication_save.pdf").exists()


# Performance tests
class TestPerformance:
    """Test performance characteristics"""

    def test_large_data_handling(self, output_dir):
        """Test handling of larger datasets"""
        from ures.plot.ml_visual import MLVisualizer, MLPlotConfig

        # Create larger dataset
        n_points = 1000
        large_df = pl.DataFrame(
            {
                "x": np.random.normal(0, 1, n_points),
                "y": np.random.normal(0, 1, n_points),
                "category": np.random.choice(["A", "B", "C"], n_points),
            }
        )

        viz = MLVisualizer(mode="publication")
        config = MLPlotConfig(
            title="Large Data Test",
            large_data_threshold=500,  # Lower threshold for testing
        )

        # Should handle large data gracefully
        fig = viz.large_scatter(large_df, config, "x", "y", color_col="category")

        assert fig is not None
        if hasattr(fig, "savefig"):
            fig.savefig(output_dir / "test_large_data.png")
            plt.close(fig)


# Parametrized tests
class TestParametrized:
    """Parametrized tests for different configurations"""

    @pytest.mark.parametrize("mode", ["exploration", "publication"])
    def test_different_modes(self, mode, sample_data):
        """Test plots in different modes"""
        from ures.plot.ml_visual import MLVisualizer, MLPlotConfig

        viz = MLVisualizer(mode=mode)
        config = MLPlotConfig(title=f"Mode Test - {mode}")
        df = sample_data["statistical"]

        fig = viz.scatter_plot(df, config, "feature1", "feature2")
        assert fig is not None
        if hasattr(fig, "savefig"):
            plt.close(fig)

    @pytest.mark.parametrize(
        "plot_type,columns",
        [
            ("scatter", ("feature1", "feature2")),
            ("line", ("feature1", "feature2")),
            ("bar", ("group", "feature1")),
        ],
    )
    def test_different_plot_types(self, plot_type, columns, sample_data):
        """Test different plot types"""
        from ures.plot.ml_visual import MLVisualizer, MLPlotConfig

        viz = MLVisualizer(mode="publication")
        config = MLPlotConfig(title=f"Plot Type Test - {plot_type}")
        df = sample_data["statistical"]

        if plot_type == "scatter":
            fig = viz.scatter_plot(df, config, columns[0], columns[1])
        elif plot_type == "line":
            fig = viz.line_plot(df, config, columns[0], columns[1])
        elif plot_type == "bar":
            fig = viz.bar_plot(df, config, columns[0], columns[1])

        assert fig is not None
        if hasattr(fig, "savefig"):
            plt.close(fig)


# Cleanup
@pytest.fixture(autouse=True)
def cleanup_plots():
    """Cleanup matplotlib figures after each test"""
    yield
    plt.close("all")


# Test markers for different categories
pytestmark = [pytest.mark.ml_visual, pytest.mark.plotting]


# Optional dependency tests
@pytest.mark.skipif(
    not pytest.importorskip("seaborn", reason="seaborn not available"),
    reason="Skip seaborn tests if not available",
)
class TestSeabornIntegration:
    """Test seaborn integration if available"""

    def test_seaborn_styling(self, sample_data, output_dir):
        """Test seaborn styling integration"""
        from ures.plot.ml_visual import MLVisualizer, MLPlotConfig

        viz = MLVisualizer(mode="publication")
        config = MLPlotConfig(title="Seaborn Test")
        df = sample_data["statistical"]

        fig = viz.box_plot(df, config, "feature2", "group")
        assert fig is not None
        if hasattr(fig, "savefig"):
            fig.savefig(output_dir / "test_seaborn.png")
            plt.close(fig)
