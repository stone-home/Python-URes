import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ures.plot.chart import ExtendedResearchPlotter, PlotConfig
from ures.plot.utils import ColorScheme, ColorConfig
from ures.plot.ml_visual import MLVisualizer, MLPlotConfig


# ============================================================================
# USAGE EXAMPLES for Chart.py
# ============================================================================


# Example usage and demonstrations
def comprehensive_example():
    """Comprehensive example showing all plot types"""
    # Generate sample data
    np.random.seed(42)
    n = 200

    # Basic data
    df_basic = pl.DataFrame(
        {
            "x": np.linspace(0, 10, n),
            "y": np.sin(np.linspace(0, 10, n)) + np.random.normal(0, 0.1, n),
            "z": np.cos(np.linspace(0, 10, n)) + np.random.normal(0, 0.1, n),
            "category": np.random.choice(["A", "B", "C"], n),
            "size": np.random.uniform(10, 100, n),
            "error": np.random.uniform(0.05, 0.2, n),
        }
    )

    # Time series data
    dates = pd.date_range("2020-01-01", periods=365, freq="D")
    trend = np.linspace(100, 200, 365)
    seasonal = 10 * np.sin(
        2 * np.pi * np.arange(365) / (365.25 / 4)
    )  # Quarterly pattern
    noise = np.random.normal(0, 5, 365)
    ts_values = trend + seasonal + noise

    df_ts = pl.DataFrame({"date": dates, "value": ts_values})

    # Scientific data
    concentrations = np.logspace(-3, 2, 50)
    responses = 100 / (1 + (10 / concentrations) ** 1.5) + np.random.normal(0, 3, 50)
    predicted = 100 / (1 + (10 / concentrations) ** 1.5)

    df_sci = pl.DataFrame(
        {
            "concentration": concentrations,
            "response": responses,
            "predicted": predicted,
            "log_conc": np.log10(concentrations),
        }
    )

    # Multi-dimensional data
    df_multi = pl.DataFrame(
        {
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(0, 1, 100),
            "feature3": np.random.normal(0, 1, 100),
            "feature4": np.random.normal(0, 1, 100),
            "class": np.random.choice(["Type1", "Type2"], 100),
        }
    )

    # Create plotter
    plotter = ExtendedResearchPlotter("matplotlib")

    # Basic configuration
    config = PlotConfig(
        title="Research Plot Example",
        xlabel="X Axis",
        ylabel="Y Axis",
        figsize=(10, 6),
        color_config=ColorConfig(scheme=ColorScheme.SCIENTIFIC),
    )

    # Demonstrate all plot types
    plot_examples = {
        # Basic Statistical Charts
        "line": lambda: plotter.line_plot(df_basic, config, "x", "y"),
        "scatter": lambda: plotter.scatter_plot(df_basic, config, "x", "y"),
        "bar": lambda: plotter.bar_plot(
            df_basic.group_by("category").agg(pl.col("y").mean()),
            config,
            "category",
            "y",
        ),
        "histogram": lambda: plotter.histogram(df_basic, config, "y"),
        "box": lambda: plotter.box_plot(df_basic, config, "y", "category"),
        "violin": lambda: plotter.violin_plot(df_basic, config, "y", "category"),
        # Multi-dimensional
        "heatmap": lambda: plotter.heatmap(
            df_multi.select(["feature1", "feature2", "feature3", "feature4"]),
            config,
            correlation=True,
        ),
        "bubble": lambda: plotter.bubble_plot(df_basic, config, "x", "y", "size"),
        "radar": lambda: plotter.radar_chart(
            df_multi.group_by("class").agg(
                [
                    pl.col("feature1").mean().alias("feature1"),
                    pl.col("feature2").mean().alias("feature2"),
                    pl.col("feature3").mean().alias("feature3"),
                    pl.col("feature4").mean().alias("feature4"),
                ]
            ),
            config,
            "class",
            ["feature1", "feature2", "feature3", "feature4"],
        ),
        "parallel": lambda: plotter.parallel_coordinates(
            df_multi, config, ["feature1", "feature2", "feature3", "feature4"], "class"
        ),
        # Scientific
        "error_bar": lambda: plotter.error_bar_plot(
            df_basic, config, "x", "y", "error"
        ),
        "regression": lambda: plotter.regression_plot(
            df_sci, config, "log_conc", "response"
        ),
        "residual": lambda: plotter.residual_plot(
            df_sci, config, "log_conc", "response", "predicted"
        ),
        "qq": lambda: plotter.qq_plot(df_basic, config, "y"),
        "density": lambda: plotter.density_plot(df_basic, config, "y"),
        # Multi-subplot
        "pair": lambda: plotter.pair_plot(
            df_multi, config, ["feature1", "feature2", "feature3"]
        ),
        "facet": lambda: plotter.facet_grid(
            df_basic, config, "scatter", "x", "y", "category"
        ),
        # Time series
        "ts_decomp": lambda: plotter.time_series_decomposition(
            df_ts, config, "date", "value", period=91
        ),
        # Must provide period
        "acf": lambda: plotter.acf_plot(
            df_ts, config, "value"
        ),  # Renamed from acf_pacf
    }

    # Generate all plots
    for plot_name, plot_func in plot_examples.items():
        save_dir = Path(__file__).parent.joinpath("images")
        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            print(f"Generating {plot_name} plot...")
            fig = plot_func()
            plotter.export(
                fig, str(save_dir.joinpath(f"{plot_name}_example.png")), config
            )
            print(f"--> Saved {plot_name}_example.png")
            if hasattr(fig, "clear"):
                fig.clear()
            if hasattr(fig, "close"):
                plt.close(fig)
        except Exception as e:
            print(f"Error generating {plot_name}: {e}")
            plt.close("all")  # Close any lingering plots on error


# ============================================================================
# USAGE EXAMPLES AND DEMO DATA GENERATORS
# ============================================================================


def generate_ml_demo_data():
    """Generate demo data for ML visualization examples"""
    np.random.seed(42)

    # Training curves data
    epochs = 100
    train_loss = 2.0 * np.exp(-np.arange(epochs) * 0.05) + np.random.normal(
        0, 0.1, epochs
    )
    val_loss = 2.2 * np.exp(-np.arange(epochs) * 0.04) + np.random.normal(
        0, 0.15, epochs
    )

    training_data = pl.DataFrame(
        {
            "epoch": np.arange(epochs),
            "train_loss": np.maximum(train_loss, 0.1),
            "val_loss": np.maximum(val_loss, 0.1),
            "train_acc": 1 - np.maximum(train_loss, 0.1) / 2,
            "val_acc": 1 - np.maximum(val_loss, 0.1) / 2,
        }
    )

    # Model comparison data
    models = ["CNN", "ResNet", "Transformer", "SVM", "Random Forest"]
    accuracies = [0.92, 0.95, 0.97, 0.88, 0.90]
    errors = [0.02, 0.015, 0.01, 0.03, 0.025]

    model_data = pl.DataFrame(
        {"model": models, "accuracy": accuracies, "std_error": errors}
    )

    # Large dataset for scatter plot
    n_large = 100000
    large_data = pl.DataFrame(
        {
            "feature1": np.random.normal(0, 1, n_large),
            "feature2": np.random.normal(0, 1, n_large),
            "cluster": np.random.choice(["A", "B", "C"], n_large),
        }
    )

    # Hyperparameter optimization data
    learning_rates = [0.001, 0.01, 0.1, 0.3]
    batch_sizes = [16, 32, 64, 128]

    hyperparam_data = []
    for lr in learning_rates:
        for bs in batch_sizes:
            # Simulate accuracy based on hyperparameters
            acc = (
                0.9
                - abs(np.log10(lr) + 2) * 0.1
                - abs(np.log2(bs) - 6) * 0.02
                + np.random.normal(0, 0.01)
            )
            hyperparam_data.append(
                {"learning_rate": lr, "batch_size": bs, "accuracy": acc}
            )

    hyperparam_df = pl.DataFrame(hyperparam_data)

    return {
        "training": training_data,
        "models": model_data,
        "large_scatter": large_data,
        "hyperparams": hyperparam_df,
    }


def demo_ml_workflow():
    """Demonstrate the dual-track ML visualization workflow"""
    print("🔬 ML Dual-Track Visualization Demo")
    print("=" * 50)

    # Generate demo data
    data = generate_ml_demo_data()

    # ========================================================================
    # EXPLORATION PHASE
    # ========================================================================
    print("\n📊 EXPLORATION PHASE (Fast & Interactive)")
    print("-" * 40)

    try:
        # Create exploration visualizer
        explorer = MLVisualizer(mode="exploration")

        # Exploration config
        explore_config = MLPlotConfig(
            title="Training Progress Exploration",
            xlabel="Epoch",
            ylabel="Loss",
            mode="exploration",
        )

        # Fast training curves exploration
        print("✓ Creating interactive training curves...")
        training_plot = explorer.training_curves(
            data["training"], explore_config, y_cols=["train_loss", "val_loss"]
        )

        # Large dataset exploration
        print("✓ Creating large dataset scatter plot...")
        scatter_config = MLPlotConfig(
            title="Feature Space Exploration",
            xlabel="Feature 1",
            ylabel="Feature 2",
            mode="exploration",
        )

        large_scatter = explorer.large_scatter(
            data["large_scatter"],
            scatter_config,
            "feature1",
            "feature2",
            color_col="cluster",
        )

        # Hyperparameter exploration
        print("✓ Creating hyperparameter heatmap...")
        hyperparam_config = MLPlotConfig(
            title="Hyperparameter Optimization", mode="exploration"
        )

        heatmap = explorer.hyperparameter_heatmap(
            data["hyperparams"],
            hyperparam_config,
            "learning_rate",
            "batch_size",
            "accuracy",
        )

    except ImportError:
        print("⚠️  HoloViews not available. Skipping exploration phase.")

    # ========================================================================
    # PUBLICATION PHASE
    # ========================================================================
    print("\n📄 PUBLICATION PHASE (High Quality)")
    print("-" * 40)

    # Create publication visualizer
    publisher = MLVisualizer(mode="publication")

    # Publication config
    pub_config = MLPlotConfig(
        title="Training Curves",
        xlabel="Epoch",
        ylabel="Loss",
        mode="publication",
        figsize=(3.5, 2.5),  # Single column width
    )

    # Add annotations using existing system
    pub_config.annotations.add_text("Convergence", 60, 0.3, fontsize=8, color="red")
    pub_config.annotations.add_arrow("Overfitting", 80, 0.5, 85, 0.7, color="blue")

    # Publication training curves
    print("✓ Creating publication training curves...")
    pub_training = publisher.training_curves(
        data["training"], pub_config, y_cols=["train_loss", "val_loss"]
    )

    # Model comparison for publication
    model_config = MLPlotConfig(
        title="Model Performance Comparison",
        xlabel="Model",
        ylabel="Accuracy",
        mode="publication",
        figsize=(3.5, 2.5),
    )

    print("✓ Creating publication model comparison...")
    pub_models = publisher.model_comparison(
        data["models"], model_config, "model", "accuracy", "std_error"
    )

    # Save publication figures
    print("✓ Saving publication figures...")
    save_dir = Path(__file__).parent.joinpath("images")
    save_dir.mkdir(parents=True, exist_ok=True)
    try:
        publisher.save_publication_figure(
            pub_training, save_dir.joinpath("training_curves_paper"), pub_config
        )
        publisher.save_publication_figure(
            pub_models, save_dir.joinpath("model_comparison_paper"), model_config
        )
        print("✓ Figures saved successfully!")
    except Exception as e:
        print(f"⚠️  Could not save figures: {e}")

    print("\n🎉 Demo completed!")
    print("\nWorkflow Summary:")
    print("1. 📊 Exploration: Fast interactive plots for data discovery")
    print("2. 📄 Publication: High-quality figures for papers")
    print("3. 🔄 Seamless transition between modes")
    print("4. 📝 Reused annotation and configuration system")


if __name__ == "__main__":
    demo_ml_workflow()
    comprehensive_example()
