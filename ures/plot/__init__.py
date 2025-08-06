import polars as pl
import numpy as np
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from .backend import (
    ResearchPlotter,
    Annotation,
    AnnotationCollection,
    PlotBackend,
    PlotConfig,
    ColorConfig,
    ColorScheme,
)
from .ml import (
    HoloViewsMLBackend,
    MatplotlibMLBackend,
    MLPlotConfig,
    HAS_HOLOVIEWS,
    HAS_DATASHADER,
    HAS_SEABORN,
    HAS_SKLEARN,
)


# ============================================================================
# UNIFIED ML VISUALIZER
# ============================================================================


class MLVisualizer:
    """Unified ML visualizer with dual-backend support"""

    def __init__(self, mode: str = "exploration"):
        """
        Initialize ML visualizer

        Args:
                mode: "exploration" for interactive plots, "publication" for high-quality plots
        """
        self.mode = mode
        self._setup_backend()

    def _setup_backend(self):
        """Setup backend with fallback strategy"""
        if self.mode == "exploration":
            if HAS_HOLOVIEWS:
                self.backend = HoloViewsMLBackend()
                self.backend_name = "holoviews"
            else:
                warnings.warn(
                    "HoloViews not available. Falling back to publication mode."
                )
                self.backend = MatplotlibMLBackend()
                self.backend_name = "matplotlib_fallback"
                self.mode = "publication"
        else:  # publication mode
            self.backend = MatplotlibMLBackend()
            self.backend_name = "matplotlib"

    def switch_mode(self, mode: str):
        """Switch between exploration and publication modes"""
        self.mode = mode
        self._setup_backend()

    # Core plotting methods
    def line_plot(
        self, df: pl.DataFrame, config: MLPlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Create line plot"""
        return self.backend.create_line_plot(df, config, x_col, y_col, **kwargs)

    def scatter_plot(
        self, df: pl.DataFrame, config: MLPlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Create scatter plot"""
        return self.backend.create_scatter_plot(df, config, x_col, y_col, **kwargs)

    def bar_plot(
        self, df: pl.DataFrame, config: MLPlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Create bar plot"""
        return self.backend.create_bar_plot(df, config, x_col, y_col, **kwargs)

    def histogram(self, df: pl.DataFrame, config: MLPlotConfig, col: str, **kwargs):
        """Create histogram"""
        return self.backend.create_histogram(df, config, col, **kwargs)

    # ML-specific methods
    def training_curves(
        self,
        df: pl.DataFrame,
        config: MLPlotConfig,
        x_col: str = "epoch",
        y_cols: List[str] = ["train_loss", "val_loss"],
        **kwargs,
    ):
        """Create training curves visualization"""
        return self.backend.create_training_curves(df, config, x_col, y_cols, **kwargs)

    def confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        config: MLPlotConfig,
        class_names: Optional[List[str]] = None,
    ):
        """Create confusion matrix heatmap"""
        return self.backend.create_confusion_matrix(y_true, y_pred, config, class_names)

    def feature_importance(
        self, importance_data: Dict[str, float], config: MLPlotConfig, top_n: int = 20
    ):
        """Create feature importance plot"""
        return self.backend.create_feature_importance(importance_data, config, top_n)

    def roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        config: MLPlotConfig,
        pos_label: int = 1,
    ):
        """Create ROC curve (publication mode only)"""
        if hasattr(self.backend, "create_roc_curve"):
            return self.backend.create_roc_curve(y_true, y_scores, config, pos_label)
        else:
            warnings.warn("ROC curve only available in publication mode")
            return None

    def large_scatter(
        self,
        df: pl.DataFrame,
        config: MLPlotConfig,
        x_col: str,
        y_col: str,
        color_col: Optional[str] = None,
        **kwargs,
    ):
        """Optimized scatter plot for large datasets"""
        if self.mode == "exploration" and HAS_HOLOVIEWS:
            # Use datashading for large datasets
            return self.backend.create_scatter_plot(
                df, config, x_col, y_col, color_col=color_col, **kwargs
            )
        else:
            # Sample for publication plots
            if len(df) > config.max_sample_size:
                df_sampled = df.sample(n=config.max_sample_size, seed=42)
                warnings.warn(f"Sampled {len(df_sampled)} points from {len(df)}")
                return self.backend.create_scatter_plot(
                    df_sampled, config, x_col, y_col, **kwargs
                )
            else:
                return self.backend.create_scatter_plot(
                    df, config, x_col, y_col, **kwargs
                )

    # Utility methods
    def save_figure(self, fig, filepath: Union[str, Path], config: MLPlotConfig):
        """Save figure with appropriate settings"""
        filepath = str(filepath)

        if self.mode == "publication":
            pub_config = config.for_publication()

            # Ensure proper extension
            if not any(
                filepath.endswith(ext) for ext in [".pdf", ".eps", ".png", ".tiff"]
            ):
                filepath += f".{pub_config.export_format}"

            if hasattr(fig, "savefig"):  # matplotlib figure
                fig.savefig(
                    filepath,
                    format=pub_config.export_format,
                    dpi=pub_config.export_dpi,
                    bbox_inches="tight",
                    pad_inches=0.05,
                )
            else:
                self.backend.export_plot(fig, filepath, pub_config)
        else:
            # Export exploration plots
            self.backend.export_plot(fig, filepath, config)

    def get_backend_info(self) -> Dict[str, Any]:
        """Get current backend information"""
        return {
            "mode": self.mode,
            "backend": self.backend_name,
            "backend_type": type(self.backend).__name__,
            "has_holoviews": HAS_HOLOVIEWS,
            "has_datashader": HAS_DATASHADER,
            "has_seaborn": HAS_SEABORN,
            "has_sklearn": HAS_SKLEARN,
        }


# ============================================================================
# CONVENIENCE FUNCTIONS FOR QUICK PLOTTING
# ============================================================================


def quick_training_plot(df: pl.DataFrame, title: str = "Training Curves", **kwargs):
    """Quick training curves plot"""
    config = MLPlotConfig(title=title, mode="exploration")
    visualizer = MLVisualizer("exploration")
    return visualizer.training_curves(df, config, **kwargs)


def quick_scatter(
    df: pl.DataFrame, x_col: str, y_col: str, title: str = "Scatter Plot", **kwargs
):
    """Quick scatter plot with auto-optimization for large data"""
    config = MLPlotConfig(title=title, xlabel=x_col, ylabel=y_col, mode="exploration")
    visualizer = MLVisualizer("exploration")
    return visualizer.large_scatter(df, config, x_col, y_col, **kwargs)


def quick_feature_importance(
    importance_dict: Dict[str, float], title: str = "Feature Importance", **kwargs
):
    """Quick feature importance plot"""
    config = MLPlotConfig(title=title, mode="publication")
    visualizer = MLVisualizer("publication")
    return visualizer.feature_importance(importance_dict, config, **kwargs)


def quick_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    **kwargs,
):
    """Quick confusion matrix plot"""
    config = MLPlotConfig(title=title, mode="publication")
    visualizer = MLVisualizer("publication")
    return visualizer.confusion_matrix(y_true, y_pred, config, class_names, **kwargs)


def publication_ready(plot_func, *args, **kwargs):
    """Convert any plot to publication-ready format"""
    # Extract config if provided
    config = None
    if args and isinstance(args[1], MLPlotConfig):
        config = args[1].for_publication()
        args = list(args)
        args[1] = config

    # Switch to publication mode temporarily
    original_mode = kwargs.get("mode", "exploration")
    kwargs["mode"] = "publication"

    try:
        return plot_func(*args, **kwargs)
    finally:
        kwargs["mode"] = original_mode
