"""
ML Dual-Track Visualization System - Reusing Existing Utils
Exploration: HoloViews + Datashader (Speed + Interactivity)
Publication: Matplotlib + Seaborn (Quality + Control)
"""

import numpy as np
import polars as pl
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field

# Core dependencies
import matplotlib.pyplot as plt

# Reuse existing annotation and configuration classes
from ures.plot.utils import (
    PlotConfig,
    ColorConfig,
    ColorScheme,
    PlotBackend,
    MatplotlibBackend,
    ResearchPlotter,
)

# Optional exploration dependencies
try:
    import holoviews as hv
    import panel as pn
    import datashader as ds
    from holoviews.operation.datashader import datashade, shade, dynspread

    HAS_HOLOVIEWS = True
    # Set backend
    hv.extension("bokeh", "matplotlib")
except ImportError:
    HAS_HOLOVIEWS = False
    warnings.warn("HoloViews not available. Exploration features limited.")

# Optional publication dependencies
try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    warnings.warn("Seaborn not available. Publication styling limited.")

try:
    from scipy import stats
    from scipy.interpolate import interp1d

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import acf, pacf

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# Optional ML-specific dependencies
try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("Scikit-learn not available. ML-specific plots limited.")


# ============================================================================
# EXTEND EXISTING CONFIG FOR ML USE CASES
# ============================================================================


@dataclass
class MLPlotConfig(PlotConfig):
    """Extended ML configuration based on existing PlotConfig"""

    # ML-specific settings
    experiment_name: str = "experiment"
    metric_name: str = "accuracy"

    # Exploration vs Publication mode
    mode: str = "exploration"  # "exploration" or "publication"

    # Large data handling
    datashade_large_data: bool = True
    large_data_threshold: int = 50000

    # Publication-specific overrides
    publication_dpi: int = 300
    publication_format: str = "pdf"

    def for_publication(self) -> "MLPlotConfig":
        """Create publication-ready version of config"""
        pub_config = MLPlotConfig(
            # Copy base settings
            figsize=self.figsize,
            title=self.title,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            x_unit=self.x_unit,
            y_unit=self.y_unit,
            # Override for publication
            mode="publication",
            style="seaborn-v0_8-paper" if HAS_SEABORN else "default",
            export_dpi=self.publication_dpi,
            export_format=self.publication_format,
            smooth=False,  # Disable smoothing for precise publication plots
            # Publication-friendly colors
            color_config=ColorConfig(scheme=ColorScheme.COLORBLIND_FRIENDLY),
            # Publication fonts
            font_config={
                "title_size": 10,
                "label_size": 8,
                "tick_size": 7,
                "legend_size": 7,
            },
            # Copy ML-specific settings
            experiment_name=self.experiment_name,
            metric_name=self.metric_name,
        )
        return pub_config


@dataclass
class PaperStyleConfig:
    """IEEE/ACM publication style configuration"""

    # Figure dimensions (inches) - IEEE standard
    single_column_width: float = 3.5
    double_column_width: float = 7.0
    max_height: float = 9.0

    # Typography
    font_family: str = "Arial"
    base_font_size: int = 8

    # Line and marker specs
    line_width: float = 1.0
    marker_size: float = 4.0

    # Colors (colorblind + print friendly)
    colors: List[str] = field(
        default_factory=lambda: [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
    )


# ============================================================================
# EXPLORATION BACKEND - EXTENDING EXISTING BACKEND INTERFACE
# ============================================================================


class ExplorationBackend(PlotBackend):
    """HoloViews-based backend for fast exploration"""

    def __init__(self):
        if not HAS_HOLOVIEWS:
            raise ImportError("HoloViews required for exploration backend")

        # Configure HoloViews defaults
        hv.opts.defaults(
            hv.opts.Curve(width=800, height=400, tools=["hover"]),
            hv.opts.Scatter(width=800, height=600, tools=["hover", "box_select"]),
            hv.opts.HeatMap(width=600, height=400, colorbar=True, tools=["hover"]),
        )

    def create_training_curves(
        self,
        df: pl.DataFrame,
        config: MLPlotConfig,
        x_col: str = "epoch",
        y_cols: List[str] = ["train_loss", "val_loss"],
        **kwargs,
    ):
        """Training curves with mode-aware rendering"""
        # Fast interactive curves
        curves = []
        colors = ["blue", "red", "green", "orange", "purple"]
        for i, y_col in enumerate(y_cols):
            curve = hv.Curve(df.to_pandas(), x_col, y_col, label=y_col)
            curve = curve.opts(color=colors[i % len(colors)], line_width=2)
            curves.append(curve)

        overlay = hv.Overlay(curves).opts(
            title=config.title,
            xlabel=config.xlabel or x_col,
            ylabel=config.ylabel or "Value",
            width=800,
            height=400,
            legend_position="right",
        )
        return overlay

    def create_model_comparison(
        self,
        df: pl.DataFrame,
        config: MLPlotConfig,
        model_col: str,
        metric_col: str,
        error_col: Optional[str] = None,
        **kwargs,
    ):
        """Model comparison bar chart"""
        # Interactive bar chart
        bars = hv.Bars(df.to_pandas(), model_col, metric_col)
        return bars.opts(
            title=config.title,
            xlabel=config.xlabel or model_col,
            ylabel=config.ylabel or metric_col,
            width=600,
            height=400,
            tools=["hover"],
        )

    def create_hyperparameter_heatmap(
        self,
        df: pl.DataFrame,
        config: MLPlotConfig,
        x_param: str,
        y_param: str,
        metric: str = "accuracy",
        **kwargs,
    ):
        """Hyperparameter optimization heatmap"""
        # Interactive heatmap
        heatmap = hv.HeatMap(df.to_pandas(), [x_param, y_param], metric)
        return heatmap.opts(
            title=f"{config.title}: {metric}",
            xlabel=x_param,
            ylabel=y_param,
            width=600,
            height=400,
            colorbar=True,
            tools=["hover"],
        )

    def create_line_plot(
        self, df: pl.DataFrame, config: MLPlotConfig, x_col: str, y_col: str, **kwargs
    ) -> "hv.Element":
        """Fast interactive line plot"""
        curve = hv.Curve(df.to_pandas(), x_col, y_col, label=config.title)

        return curve.opts(
            title=config.title,
            xlabel=(
                f"{config.xlabel} ({config.x_unit})" if config.x_unit else config.xlabel
            ),
            ylabel=(
                f"{config.ylabel} ({config.y_unit})" if config.y_unit else config.ylabel
            ),
            width=800,
            height=400,
            tools=["hover", "box_zoom", "reset"],
        )

    def create_scatter_plot(
        self, df: pl.DataFrame, config: MLPlotConfig, x_col: str, y_col: str, **kwargs
    ) -> "hv.Element":
        """High-performance scatter plot with automatic datashading"""
        color_col = kwargs.get("color_col", None)

        if color_col:
            points = hv.Points(df.to_pandas(), [x_col, y_col], color_col)
        else:
            points = hv.Points(df.to_pandas(), [x_col, y_col])

        # Auto-datashade for large datasets
        if len(df) > config.large_data_threshold and config.datashade_large_data:
            return datashade(points, cmap="viridis", width=800, height=600)
        else:
            return points.opts(
                title=config.title,
                xlabel=(
                    f"{config.xlabel} ({config.x_unit})"
                    if config.x_unit
                    else config.xlabel
                ),
                ylabel=(
                    f"{config.ylabel} ({config.y_unit})"
                    if config.y_unit
                    else config.ylabel
                ),
                color=color_col if color_col else "blue",
                size=5,
                alpha=0.6,
                width=800,
                height=600,
                tools=["hover", "box_select", "lasso_select"],
            )

    def apply_annotations(self, fig, config: MLPlotConfig):
        """Apply annotations to HoloViews plot (simplified)"""
        # HoloViews handles annotations differently
        # This is a placeholder for HoloViews annotation system
        pass

    def export_plot(self, fig, filepath: str, config: MLPlotConfig):
        """Export HoloViews plot"""
        try:
            hv.save(fig, filepath)
        except Exception as e:
            warnings.warn(f"Failed to export HoloViews plot: {e}")


# ============================================================================
# PUBLICATION BACKEND - EXTENDING EXISTING MATPLOTLIB BACKEND
# ============================================================================


class PublicationBackend(MatplotlibBackend):
    """Enhanced matplotlib backend for publication-quality plots"""

    def __init__(self, paper_style: Optional[PaperStyleConfig] = None):
        super().__init__()
        self.paper_style = paper_style or PaperStyleConfig()
        self._setup_publication_style()

    def _setup_publication_style(self):
        """Configure matplotlib for publication quality"""
        plt.rcParams.update(
            {
                "font.family": self.paper_style.font_family,
                "font.size": self.paper_style.base_font_size,
                "axes.linewidth": 0.5,
                "lines.linewidth": self.paper_style.line_width,
                "patch.linewidth": 0.5,
                "axes.labelsize": self.paper_style.base_font_size,
                "axes.titlesize": self.paper_style.base_font_size + 2,
                "xtick.labelsize": self.paper_style.base_font_size - 1,
                "ytick.labelsize": self.paper_style.base_font_size - 1,
                "legend.fontsize": self.paper_style.base_font_size - 1,
                "figure.titlesize": self.paper_style.base_font_size + 4,
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.05,
            }
        )

        if HAS_SEABORN:
            sns.set_palette("colorblind")

    def _setup_figure(self, config: MLPlotConfig):
        """Override to use publication-specific figure setup"""
        if config.mode == "publication":
            # Use paper-specific figure size
            if config.figsize[0] > 4:  # Assume double column if width > 4
                figsize = (self.paper_style.double_column_width, config.figsize[1])
            else:
                figsize = (self.paper_style.single_column_width, config.figsize[1])
        else:
            figsize = config.figsize

        # Use available styles with publication preference
        if config.mode == "publication":
            style = config.style
        else:
            style = "default"  # Keep simple for exploration

        try:
            plt.style.use(style)
        except:
            plt.style.use("default")

        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=config.dpi)
        self._set_labels(config)

    def create_training_curves(
        self,
        df: pl.DataFrame,
        config: MLPlotConfig,
        x_col: str = "epoch",
        y_cols: List[str] = ["train_loss", "val_loss"],
        **kwargs,
    ) -> plt.Figure:
        """Publication-ready training curves"""
        self._setup_figure(config)

        colors = self.paper_style.colors

        for i, y_col in enumerate(y_cols):
            if HAS_SEABORN:
                sns.lineplot(
                    data=df.to_pandas(),
                    x=x_col,
                    y=y_col,
                    ax=self.ax,
                    color=colors[i % len(colors)],
                    linewidth=self.paper_style.line_width,
                    label=y_col,
                )
            else:
                self.ax.plot(
                    df[x_col],
                    df[y_col],
                    color=colors[i % len(colors)],
                    linewidth=self.paper_style.line_width,
                    label=y_col,
                )

        self.ax.set_xlabel(config.xlabel or x_col.replace("_", " ").title())
        self.ax.set_ylabel(config.ylabel or "Loss")
        self.ax.set_title(config.title)
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

        # Apply existing annotation system
        self.apply_annotations(self.fig, config)

        return self.fig

    def create_model_comparison(
        self,
        df: pl.DataFrame,
        config: MLPlotConfig,
        model_col: str,
        metric_col: str,
        error_col: Optional[str] = None,
        **kwargs,
    ) -> plt.Figure:
        """Publication-ready model comparison bar chart"""
        self._setup_figure(config)

        if HAS_SEABORN:
            sns.barplot(
                data=df.to_pandas(),
                x=model_col,
                y=metric_col,
                hue=model_col,  # Assign the x-variable to hue
                legend=False,  # Hide the legend to mimic old behavior
                ax=self.ax,
                palette="colorblind",
                errorbar=None,
            )

            # Add error bars manually if provided
            if error_col:
                x_pos = range(len(df[model_col].unique()))
                self.ax.errorbar(
                    x_pos,
                    df[metric_col],
                    yerr=df[error_col],
                    fmt="none",
                    ecolor="black",
                    capsize=3,
                )
        else:
            self.ax.bar(
                df[model_col],
                df[metric_col],
                color=self.paper_style.colors[0],
                yerr=df[error_col] if error_col else None,
                capsize=3,
            )

        self.ax.set_xlabel(config.xlabel or model_col.replace("_", " ").title())
        self.ax.set_ylabel(config.ylabel or metric_col.replace("_", " ").title())
        self.ax.set_title(config.title)
        self.ax.grid(True, alpha=0.3)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")

        # Apply existing annotation system
        self.apply_annotations(self.fig, config)

        return self.fig

    # ========================================================================
    # BASIC STATISTICAL CHARTS
    # ========================================================================

    def create_line_plot(
        self, df: pl.DataFrame, config: MLPlotConfig, x_col: str, y_col: str, **kwargs
    ) -> plt.Figure:
        """Publication line plot - time series data, trend analysis"""
        self._setup_figure(config)

        colors = self.paper_style.colors

        # Handle multiple series
        if "group_col" in kwargs:
            group_col = kwargs["group_col"]
            for i, group in enumerate(df[group_col].unique()):
                group_data = df.filter(pl.col(group_col) == group)
                self.ax.plot(
                    group_data[x_col],
                    group_data[y_col],
                    color=colors[i % len(colors)],
                    linewidth=self.paper_style.line_width,
                    label=str(group),
                )
            if config.legend:
                self.ax.legend()
        else:
            self.ax.plot(
                df[x_col],
                df[y_col],
                color=colors[0],
                linewidth=self.paper_style.line_width,
            )

        self.ax.grid(True, alpha=0.3)
        self.apply_annotations(self.fig, config)
        return self.fig

    def create_scatter_plot(
        self, df: pl.DataFrame, config: MLPlotConfig, x_col: str, y_col: str, **kwargs
    ) -> plt.Figure:
        """Publication scatter plot - correlation analysis, data distribution"""
        self._setup_figure(config)

        colors = self.paper_style.colors
        size_col = kwargs.get("size_col", None)
        color_col = kwargs.get("color_col", None)

        scatter_kwargs = {
            "s": self.paper_style.marker_size**2,
            "alpha": kwargs.get("alpha", 0.7),
            "edgecolors": "none",
        }

        if color_col:
            # Categorical coloring
            if df[color_col].dtype == pl.Utf8:
                unique_vals = df[color_col].unique()
                for i, val in enumerate(unique_vals):
                    mask = df[color_col] == val
                    subset = df.filter(mask)
                    self.ax.scatter(
                        subset[x_col],
                        subset[y_col],
                        color=colors[i % len(colors)],
                        label=str(val),
                        **scatter_kwargs,
                    )
                if config.legend:
                    self.ax.legend()
            else:
                # Continuous coloring
                scatter = self.ax.scatter(
                    df[x_col],
                    df[y_col],
                    c=df[color_col],
                    cmap="viridis",
                    **scatter_kwargs,
                )
                plt.colorbar(scatter, ax=self.ax, label=color_col)
        else:
            self.ax.scatter(df[x_col], df[y_col], color=colors[0], **scatter_kwargs)

        self.ax.grid(True, alpha=0.3)
        self.apply_annotations(self.fig, config)
        return self.fig

    def create_bar_plot(
        self, df: pl.DataFrame, config: MLPlotConfig, x_col: str, y_col: str, **kwargs
    ) -> plt.Figure:
        """Publication bar plot - categorical data comparison"""
        self._setup_figure(config)

        error_col = kwargs.get("error_col", None)

        if HAS_SEABORN:
            sns.barplot(
                data=df.to_pandas(),
                x=x_col,
                y=y_col,
                ax=self.ax,
                palette="colorblind",
                errorbar=None,
                hue=x_col,
                legend=False,
            )
            if error_col:
                # Add error bars manually
                x_pos = range(len(df[x_col].unique()))
                self.ax.errorbar(
                    x_pos,
                    df[y_col],
                    yerr=df[error_col],
                    fmt="none",
                    ecolor="black",
                    capsize=3,
                )
        else:
            self.ax.bar(
                df[x_col],
                df[y_col],
                color=self.paper_style.colors[0],
                yerr=df[error_col] if error_col else None,
                capsize=3,
            )

        self.ax.grid(True, alpha=0.3, axis="y")
        plt.xticks(rotation=45, ha="right")
        self.apply_annotations(self.fig, config)
        return self.fig

    def create_histogram(
        self, df: pl.DataFrame, config: MLPlotConfig, col: str, **kwargs
    ) -> plt.Figure:
        """Publication histogram - data distribution, frequency analysis"""
        self._setup_figure(config)

        bins = kwargs.get("bins", 30)
        density = kwargs.get("density", False)

        self.ax.hist(
            df[col],
            bins=bins,
            density=density,
            color=self.paper_style.colors[0],
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
        )

        self.ax.set_ylabel("Density" if density else "Frequency")
        self.ax.grid(True, alpha=0.3)
        self.apply_annotations(self.fig, config)
        return self.fig

    def create_box_plot(
        self,
        df: pl.DataFrame,
        config: MLPlotConfig,
        y_col: str,
        x_col: Optional[str] = None,
        **kwargs,
    ) -> plt.Figure:
        """Publication box plot - data distribution, outlier detection"""
        self._setup_figure(config)

        if HAS_SEABORN:
            if x_col:
                sns.boxplot(
                    data=df.to_pandas(),
                    x=x_col,
                    y=y_col,
                    ax=self.ax,
                    palette="colorblind",
                )
                plt.xticks(rotation=45, ha="right")
            else:
                sns.boxplot(
                    data=df.to_pandas(),
                    y=y_col,
                    ax=self.ax,
                    color=self.paper_style.colors[0],
                )
        else:
            if x_col:
                # Grouped box plot
                data_groups = []
                labels = []
                for group in df[x_col].unique():
                    group_data = df.filter(pl.col(x_col) == group)[y_col].to_list()
                    data_groups.append(group_data)
                    labels.append(str(group))

                bp = self.ax.boxplot(data_groups, labels=labels, patch_artist=True)
                for patch, color in zip(bp["boxes"], self.paper_style.colors):
                    patch.set_facecolor(color)
                plt.xticks(rotation=45, ha="right")
            else:
                bp = self.ax.boxplot(df[y_col].to_list(), patch_artist=True)
                bp["boxes"][0].set_facecolor(self.paper_style.colors[0])

        self.ax.grid(True, alpha=0.3)
        self.apply_annotations(self.fig, config)
        return self.fig

    def create_violin_plot(
        self,
        df: pl.DataFrame,
        config: MLPlotConfig,
        y_col: str,
        x_col: Optional[str] = None,
        **kwargs,
    ) -> plt.Figure:
        """Publication violin plot - data distribution density"""
        self._setup_figure(config)

        if HAS_SEABORN:
            if x_col:
                sns.violinplot(
                    data=df.to_pandas(),
                    x=x_col,
                    y=y_col,
                    ax=self.ax,
                    palette="colorblind",
                )
                plt.xticks(rotation=45, ha="right")
            else:
                sns.violinplot(
                    data=df.to_pandas(),
                    y=y_col,
                    ax=self.ax,
                    color=self.paper_style.colors[0],
                )
        else:
            # Fallback to box plot if seaborn not available
            warnings.warn(
                "Seaborn not available. Using box plot instead of violin plot."
            )
            return self.create_box_plot(df, config, y_col, x_col, **kwargs)

        self.ax.grid(True, alpha=0.3)
        self.apply_annotations(self.fig, config)
        return self.fig

    # ========================================================================
    # MULTI-DIMENSIONAL DATA VISUALIZATION
    # ========================================================================

    def create_heatmap(
        self, df: pl.DataFrame, config: MLPlotConfig, **kwargs
    ) -> plt.Figure:
        """Publication heatmap - correlation matrix, data matrix visualization"""
        self._setup_figure(config)

        # Handle correlation matrix
        if kwargs.get("correlation", False):
            numeric_cols = [
                col
                for col in df.columns
                if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
            ]
            if HAS_PANDAS:
                corr_matrix = df.select(numeric_cols).to_pandas().corr()
            else:
                # Manual correlation calculation
                data_matrix = df.select(numeric_cols).to_numpy()
                corr_matrix = np.corrcoef(data_matrix.T)
                corr_matrix = pd.DataFrame(
                    corr_matrix, columns=numeric_cols, index=numeric_cols
                )
            data_to_plot = corr_matrix
        else:
            data_to_plot = df.to_pandas() if HAS_PANDAS else df.to_numpy()

        if HAS_SEABORN:
            sns.heatmap(
                data_to_plot,
                ax=self.ax,
                cmap="viridis",
                annot=kwargs.get("annot", True),
                fmt=kwargs.get("fmt", ".2f"),
                cbar_kws={"label": kwargs.get("cbar_label", "Value")},
            )
        else:
            im = self.ax.imshow(
                (
                    data_to_plot.values
                    if hasattr(data_to_plot, "values")
                    else data_to_plot
                ),
                cmap="viridis",
                aspect="auto",
            )
            plt.colorbar(im, ax=self.ax, label=kwargs.get("cbar_label", "Value"))

        self.apply_annotations(self.fig, config)
        return self.fig

    def create_bubble_plot(
        self,
        df: pl.DataFrame,
        config: MLPlotConfig,
        x_col: str,
        y_col: str,
        size_col: str,
        **kwargs,
    ) -> plt.Figure:
        """Publication bubble plot - three-dimensional data relationships"""
        self._setup_figure(config)

        # Normalize sizes for better visualization
        sizes = df[size_col].to_numpy()
        size_normalized = (sizes - sizes.min()) / (sizes.max() - sizes.min()) * 300 + 50

        color_col = kwargs.get("color_col", None)

        if color_col:
            if df[color_col].dtype == pl.Utf8:
                # Categorical coloring
                unique_vals = df[color_col].unique()
                for i, val in enumerate(unique_vals):
                    mask = df[color_col] == val
                    subset = df.filter(mask)
                    subset_sizes = size_normalized[df[color_col] == val]
                    self.ax.scatter(
                        subset[x_col],
                        subset[y_col],
                        s=subset_sizes,
                        color=self.paper_style.colors[i % len(self.paper_style.colors)],
                        alpha=0.6,
                        edgecolors="black",
                        linewidth=0.5,
                        label=str(val),
                    )
                if config.legend:
                    self.ax.legend()
            else:
                # Continuous coloring
                scatter = self.ax.scatter(
                    df[x_col],
                    df[y_col],
                    s=size_normalized,
                    c=df[color_col],
                    alpha=0.6,
                    cmap="viridis",
                    edgecolors="black",
                    linewidth=0.5,
                )
                plt.colorbar(scatter, ax=self.ax, label=color_col)
        else:
            self.ax.scatter(
                df[x_col],
                df[y_col],
                s=size_normalized,
                color=self.paper_style.colors[0],
                alpha=0.6,
                edgecolors="black",
                linewidth=0.5,
            )

        self.ax.grid(True, alpha=0.3)
        self.apply_annotations(self.fig, config)
        return self.fig

    def create_radar_chart(
        self,
        df: pl.DataFrame,
        config: MLPlotConfig,
        category_col: str,
        value_cols: List[str],
        **kwargs,
    ) -> plt.Figure:
        """Publication radar chart - multi-indicator comparison"""
        # Create polar subplot
        fig = plt.figure(figsize=config.figsize, dpi=config.dpi)
        ax = fig.add_subplot(111, projection="polar")
        self.fig = fig
        self.ax = ax

        angles = np.linspace(0, 2 * np.pi, len(value_cols), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        for i, category in enumerate(df[category_col].unique()):
            category_data = df.filter(pl.col(category_col) == category)
            values = [category_data[col].item() for col in value_cols]
            values += values[:1]  # Complete the circle

            color = self.paper_style.colors[i % len(self.paper_style.colors)]
            ax.plot(
                angles,
                values,
                "o-",
                linewidth=self.paper_style.line_width,
                label=str(category),
                color=color,
            )
            ax.fill(angles, values, alpha=0.25, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(value_cols)
        ax.set_title(config.title, fontsize=config.font_config["title_size"], pad=20)

        if config.legend:
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

        return fig

    def create_parallel_coordinates(
        self,
        df: pl.DataFrame,
        config: MLPlotConfig,
        value_cols: List[str],
        class_col: Optional[str] = None,
        **kwargs,
    ) -> plt.Figure:
        """Publication parallel coordinates plot - high-dimensional data visualization"""
        self._setup_figure(config)

        # Normalize data to [0, 1] for better visualization
        df_normalized = df.clone()
        for col in value_cols:
            col_data = df[col]
            min_val, max_val = col_data.min(), col_data.max()
            if max_val > min_val:
                df_normalized = df_normalized.with_columns(
                    ((pl.col(col) - min_val) / (max_val - min_val)).alias(col)
                )

        x_positions = list(range(len(value_cols)))

        if class_col:
            unique_classes = df[class_col].unique()
            for i, class_val in enumerate(unique_classes):
                class_data = df_normalized.filter(pl.col(class_col) == class_val)
                color = self.paper_style.colors[i % len(self.paper_style.colors)]

                for idx in range(len(class_data)):
                    row = class_data.row(idx)
                    y_values = [row[df.columns.index(col)] for col in value_cols]
                    self.ax.plot(
                        x_positions,
                        y_values,
                        color=color,
                        alpha=kwargs.get("alpha", 0.7),
                        linewidth=kwargs.get("linewidth", 0.5),
                    )

                # Add legend entry
                self.ax.plot(
                    [],
                    [],
                    color=color,
                    label=str(class_val),
                    linewidth=self.paper_style.line_width,
                )

            if config.legend:
                self.ax.legend()
        else:
            for idx in range(len(df_normalized)):
                row = df_normalized.row(idx)
                y_values = [row[df.columns.index(col)] for col in value_cols]
                self.ax.plot(
                    x_positions,
                    y_values,
                    color=self.paper_style.colors[0],
                    alpha=kwargs.get("alpha", 0.7),
                    linewidth=kwargs.get("linewidth", 0.5),
                )

        self.ax.set_xticks(x_positions)
        self.ax.set_xticklabels([col.replace("_", " ").title() for col in value_cols])
        self.ax.set_ylim(0, 1)
        self.ax.grid(True, alpha=0.3)
        self.apply_annotations(self.fig, config)
        return self.fig

    # ========================================================================
    # SCIENTIFIC SPECIALIZED CHARTS
    # ========================================================================

    def create_error_bar_plot(
        self,
        df: pl.DataFrame,
        config: MLPlotConfig,
        x_col: str,
        y_col: str,
        error_col: str,
        **kwargs,
    ) -> plt.Figure:
        """Publication error bar plot - experimental data uncertainty"""
        self._setup_figure(config)

        self.ax.errorbar(
            df[x_col],
            df[y_col],
            yerr=df[error_col],
            fmt="o-",
            color=self.paper_style.colors[0],
            capsize=kwargs.get("capsize", 3),
            capthick=kwargs.get("capthick", 1),
            markersize=self.paper_style.marker_size,
            linewidth=self.paper_style.line_width,
        )

        self.ax.grid(True, alpha=0.3)
        self.apply_annotations(self.fig, config)
        return self.fig

    def create_regression_plot(
        self, df: pl.DataFrame, config: MLPlotConfig, x_col: str, y_col: str, **kwargs
    ) -> plt.Figure:
        """Publication regression plot - fitting curves and confidence intervals"""
        self._setup_figure(config)

        if HAS_SCIPY:
            x_data = df[x_col].to_numpy()
            y_data = df[y_col].to_numpy()

            # Scatter plot
            self.ax.scatter(
                x_data,
                y_data,
                color=self.paper_style.colors[0],
                alpha=0.6,
                s=self.paper_style.marker_size**2,
            )

            # Regression line
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                x_data, y_data
            )
            line = slope * x_data + intercept
            self.ax.plot(
                x_data,
                line,
                color=self.paper_style.colors[1],
                linewidth=self.paper_style.line_width,
                label=f"R² = {r_value ** 2:.3f}",
            )

            # Confidence intervals
            if kwargs.get("ci", True):
                residuals = y_data - line
                mse = np.mean(residuals**2)
                n = len(x_data)
                x_mean = np.mean(x_data)
                sxx = np.sum((x_data - x_mean) ** 2)
                se = np.sqrt(mse * (1 / n + (x_data - x_mean) ** 2 / sxx))
                ci = 1.96 * se  # 95% confidence interval

                self.ax.fill_between(
                    x_data,
                    line - ci,
                    line + ci,
                    alpha=0.3,
                    color=self.paper_style.colors[1],
                )

            if config.legend:
                self.ax.legend()
        else:
            # Fallback without scipy
            if HAS_SEABORN:
                sns.regplot(
                    data=df.to_pandas(),
                    x=x_col,
                    y=y_col,
                    ax=self.ax,
                    color=self.paper_style.colors[0],
                )
            else:
                warnings.warn("Scipy not available. Creating simple scatter plot.")
                return self.create_scatter_plot(df, config, x_col, y_col, **kwargs)

        self.ax.grid(True, alpha=0.3)
        self.apply_annotations(self.fig, config)
        return self.fig

    def create_residual_plot(
        self,
        df: pl.DataFrame,
        config: MLPlotConfig,
        x_col: str,
        y_col: str,
        predicted_col: str,
        **kwargs,
    ) -> plt.Figure:
        """Publication residual plot - model diagnostics"""
        self._setup_figure(config)

        residuals = df[y_col] - df[predicted_col]

        self.ax.scatter(
            df[predicted_col],
            residuals,
            color=self.paper_style.colors[0],
            alpha=0.6,
            s=self.paper_style.marker_size**2,
        )
        self.ax.axhline(
            y=0,
            color=self.paper_style.colors[1],
            linestyle="--",
            linewidth=self.paper_style.line_width,
        )

        self.ax.set_xlabel("Predicted Values")
        self.ax.set_ylabel("Residuals")
        self.ax.grid(True, alpha=0.3)
        self.apply_annotations(self.fig, config)
        return self.fig

    def create_qq_plot(
        self, df: pl.DataFrame, config: MLPlotConfig, col: str, **kwargs
    ) -> plt.Figure:
        """Publication Q-Q plot - distribution testing"""
        self._setup_figure(config)

        if HAS_SCIPY:
            data = df[col].to_numpy()
            stats.probplot(data, dist=kwargs.get("dist", "norm"), plot=self.ax)

            # Customize colors to match paper style
            self.ax.get_lines()[0].set_markerfacecolor(self.paper_style.colors[0])
            self.ax.get_lines()[0].set_markeredgecolor(self.paper_style.colors[0])
            self.ax.get_lines()[0].set_markersize(self.paper_style.marker_size)
            self.ax.get_lines()[1].set_color(self.paper_style.colors[1])
            self.ax.get_lines()[1].set_linewidth(self.paper_style.line_width)
        else:
            warnings.warn("Scipy not available. Cannot create Q-Q plot.")
            return self.create_histogram(df, config, col, **kwargs)

        self.ax.grid(True, alpha=0.3)
        self.apply_annotations(self.fig, config)
        return self.fig

    def create_density_plot(
        self, df: pl.DataFrame, config: MLPlotConfig, col: str, **kwargs
    ) -> plt.Figure:
        """Publication density plot - probability density distribution"""
        self._setup_figure(config)

        if HAS_SCIPY:
            data = df[col].to_numpy()

            # KDE plot
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 1000)
            density = kde(x_range)

            self.ax.plot(
                x_range,
                density,
                color=self.paper_style.colors[0],
                linewidth=self.paper_style.line_width,
            )
            self.ax.fill_between(
                x_range, density, alpha=0.3, color=self.paper_style.colors[0]
            )
        elif HAS_SEABORN:
            sns.kdeplot(
                data=df.to_pandas(), x=col, ax=self.ax, color=self.paper_style.colors[0]
            )
        else:
            warnings.warn(
                "Neither Scipy nor Seaborn available. Using histogram instead."
            )
            return self.create_histogram(df, config, col, density=True, **kwargs)

        self.ax.set_ylabel("Density")
        self.ax.grid(True, alpha=0.3)
        self.apply_annotations(self.fig, config)
        return self.fig

    def create_cdf_plot(
        self, df: pl.DataFrame, config: MLPlotConfig, col: str, **kwargs
    ) -> plt.Figure:
        """Publication CDF plot - cumulative distribution function"""
        self._setup_figure(config)

        data = df[col].to_numpy()

        # Sort data for CDF
        sorted_data = np.sort(data)
        y_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

        self.ax.plot(
            sorted_data,
            y_values,
            color=self.paper_style.colors[0],
            linewidth=self.paper_style.line_width,
            drawstyle="steps-post",
        )

        # Add median line if requested
        if kwargs.get("show_median", True):
            median_val = np.median(data)
            self.ax.axvline(
                x=median_val,
                color=self.paper_style.colors[1],
                linestyle="--",
                linewidth=self.paper_style.line_width,
                label=f"Median: {median_val:.3f}",
            )
            if config.legend:
                self.ax.legend()

        self.ax.set_ylabel("Cumulative Probability")
        self.ax.set_ylim(0, 1)
        self.ax.grid(True, alpha=0.3)
        self.apply_annotations(self.fig, config)
        return self.fig

    # ========================================================================
    # MULTI-SUBPLOT COMBINATIONS
    # ========================================================================

    def create_subplots_grid(
        self, df: pl.DataFrame, config: MLPlotConfig, plot_configs: List[Dict], **kwargs
    ) -> plt.Figure:
        """Publication subplot grid - multiple related chart combinations"""
        nrows = kwargs.get("nrows", 2)
        ncols = kwargs.get("ncols", 2)

        fig, axes = plt.subplots(nrows, ncols, figsize=config.figsize, dpi=config.dpi)
        self.fig = fig

        if not isinstance(axes, np.ndarray):
            axes = [axes]
        elif axes.ndim == 2:
            axes = axes.flatten()

        for i, plot_config in enumerate(plot_configs[: len(axes)]):
            self.ax = axes[i]
            plot_type = plot_config["type"]
            plot_params = plot_config.get("params", {})

            # Create subplot based on type
            if plot_type == "line":
                self.ax.plot(
                    df[plot_params["x_col"]],
                    df[plot_params["y_col"]],
                    color=self.paper_style.colors[i % len(self.paper_style.colors)],
                )
            elif plot_type == "scatter":
                self.ax.scatter(
                    df[plot_params["x_col"]],
                    df[plot_params["y_col"]],
                    color=self.paper_style.colors[i % len(self.paper_style.colors)],
                    s=self.paper_style.marker_size**2,
                )
            elif plot_type == "hist":
                self.ax.hist(
                    df[plot_params["col"]],
                    bins=plot_params.get("bins", 30),
                    color=self.paper_style.colors[i % len(self.paper_style.colors)],
                    alpha=0.7,
                )
            elif plot_type == "box":
                if HAS_SEABORN:
                    sns.boxplot(
                        data=df.to_pandas(),
                        y=plot_params["col"],
                        ax=self.ax,
                        color=self.paper_style.colors[i % len(self.paper_style.colors)],
                    )
                else:
                    self.ax.boxplot(df[plot_params["col"]].to_list())

            self.ax.set_title(plot_config.get("title", ""))
            self.ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(len(plot_configs), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        return fig

    def create_pair_plot(
        self, df: pl.DataFrame, config: MLPlotConfig, columns: List[str], **kwargs
    ) -> plt.Figure:
        """Publication pair plot - pairwise variable relationships"""
        n_cols = len(columns)
        fig, axes = plt.subplots(n_cols, n_cols, figsize=config.figsize, dpi=config.dpi)
        self.fig = fig

        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if n_cols == 1:
                    ax = axes
                else:
                    ax = axes[i, j] if n_cols > 1 else axes[j]

                if i == j:
                    # Diagonal: histogram
                    ax.hist(
                        df[col1],
                        bins=20,
                        color=self.paper_style.colors[0],
                        alpha=0.7,
                        edgecolor="black",
                        linewidth=0.5,
                    )
                    ax.set_ylabel("Frequency")
                else:
                    # Off-diagonal: scatter plot
                    ax.scatter(
                        df[col2],
                        df[col1],
                        color=self.paper_style.colors[0],
                        alpha=0.6,
                        s=self.paper_style.marker_size**2,
                    )

                # Set labels only on edges
                if i == n_cols - 1:
                    ax.set_xlabel(col2.replace("_", " ").title())
                else:
                    ax.set_xticklabels([])
                if j == 0:
                    ax.set_ylabel(col1.replace("_", " ").title())
                else:
                    ax.set_yticklabels([])

                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_facet_grid(
        self,
        df: pl.DataFrame,
        config: MLPlotConfig,
        plot_type: str,
        x_col: str,
        y_col: str,
        facet_col: str,
        **kwargs,
    ) -> plt.Figure:
        """Publication facet grid - grouped display by category"""
        facet_values = df[facet_col].unique().to_list()
        n_facets = len(facet_values)
        ncols = kwargs.get("ncols", min(3, n_facets))
        nrows = (n_facets + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=config.figsize, dpi=config.dpi)
        self.fig = fig

        if not isinstance(axes, np.ndarray):
            axes = [axes]
        elif axes.ndim == 2:
            axes = axes.flatten()

        for i, facet_val in enumerate(facet_values):
            if i < len(axes):
                ax = axes[i]
                facet_data = df.filter(pl.col(facet_col) == facet_val)

                color = self.paper_style.colors[i % len(self.paper_style.colors)]

                if plot_type == "scatter":
                    ax.scatter(
                        facet_data[x_col],
                        facet_data[y_col],
                        color=color,
                        alpha=0.7,
                        s=self.paper_style.marker_size**2,
                    )
                elif plot_type == "line":
                    ax.plot(
                        facet_data[x_col],
                        facet_data[y_col],
                        color=color,
                        linewidth=self.paper_style.line_width,
                    )
                elif plot_type == "bar":
                    ax.bar(facet_data[x_col], facet_data[y_col], color=color, alpha=0.7)

                ax.set_title(f"{facet_col} = {facet_val}")
                ax.set_xlabel(config.xlabel or x_col.replace("_", " ").title())
                ax.set_ylabel(config.ylabel or y_col.replace("_", " ").title())
                ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(len(facet_values), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        return fig

    # ========================================================================
    # TIME SERIES SPECIALIZED
    # ========================================================================

    def create_time_series_decomposition(
        self,
        df: pl.DataFrame,
        config: MLPlotConfig,
        time_col: str,
        value_col: str,
        **kwargs,
    ) -> plt.Figure:
        """Publication time series decomposition - trend, seasonality, residual"""
        if not HAS_STATSMODELS:
            warnings.warn(
                "Statsmodels not available. Cannot create decomposition plot."
            )
            return self.create_line_plot(df, config, time_col, value_col)

        if not HAS_PANDAS:
            warnings.warn("Pandas not available. Cannot create decomposition plot.")
            return self.create_line_plot(df, config, time_col, value_col)

        # Convert to pandas for statsmodels
        df_pandas = df.select([time_col, value_col]).to_pandas()
        df_pandas[time_col] = pd.to_datetime(df_pandas[time_col])
        df_pandas.set_index(time_col, inplace=True)

        # Perform decomposition
        try:
            decomposition = seasonal_decompose(
                df_pandas[value_col],
                model=kwargs.get("model", "additive"),
                period=kwargs.get("period", None),
            )
        except ValueError as e:
            warnings.warn(f"Decomposition failed: {e}. Creating simple line plot.")
            return self.create_line_plot(df, config, time_col, value_col)

        # Create subplots
        fig, axes = plt.subplots(
            4, 1, figsize=(config.figsize[0], config.figsize[1] * 1.5)
        )
        self.fig = fig

        components = [
            ("Original", decomposition.observed),
            ("Trend", decomposition.trend),
            ("Seasonal", decomposition.seasonal),
            ("Residual", decomposition.resid),
        ]

        for i, (title, component) in enumerate(components):
            ax = axes[i]
            color = self.paper_style.colors[i % len(self.paper_style.colors)]
            ax.plot(
                component.index,
                component.values,
                color=color,
                linewidth=self.paper_style.line_width,
            )
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

            if i == len(components) - 1:  # Last subplot
                ax.set_xlabel(config.xlabel or time_col.replace("_", " ").title())

        plt.tight_layout()
        return fig

    def create_acf_pacf_plot(
        self, df: pl.DataFrame, config: MLPlotConfig, col: str, **kwargs
    ) -> plt.Figure:
        """Publication ACF/PACF plot - time series analysis"""
        if not HAS_STATSMODELS:
            warnings.warn("Statsmodels not available. Cannot create ACF/PACF plot.")
            return self.create_line_plot(df, config, "index", col)

        data = df[col].to_numpy()
        lags = kwargs.get("lags", min(40, len(data) // 4))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=config.figsize)
        self.fig = fig

        try:
            # ACF
            acf_vals = acf(data, nlags=lags, alpha=0.05)
            ax1.plot(
                acf_vals[0],
                color=self.paper_style.colors[0],
                linewidth=self.paper_style.line_width,
            )
            ax1.axhline(y=0, color="black", linestyle="-", alpha=0.3)
            ax1.set_title("Autocorrelation Function (ACF)")
            ax1.grid(True, alpha=0.3)

            # Add confidence intervals if available
            if len(acf_vals) > 1:
                ax1.fill_between(
                    range(len(acf_vals[0])),
                    acf_vals[1][:, 0],
                    acf_vals[1][:, 1],
                    alpha=0.3,
                    color=self.paper_style.colors[0],
                )

            # PACF
            pacf_vals = pacf(data, nlags=lags, alpha=0.05)
            ax2.plot(
                pacf_vals[0],
                color=self.paper_style.colors[1],
                linewidth=self.paper_style.line_width,
            )
            ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
            ax2.set_title("Partial Autocorrelation Function (PACF)")
            ax2.set_xlabel("Lag")
            ax2.grid(True, alpha=0.3)

            # Add confidence intervals if available
            if len(pacf_vals) > 1:
                ax2.fill_between(
                    range(len(pacf_vals[0])),
                    pacf_vals[1][:, 0],
                    pacf_vals[1][:, 1],
                    alpha=0.3,
                    color=self.paper_style.colors[1],
                )

        except Exception as e:
            warnings.warn(f"ACF/PACF calculation failed: {e}")
            # Fallback to simple correlation plot
            correlations = np.correlate(data, data, mode="full")
            correlations = correlations[correlations.size // 2 :]
            correlations = correlations / correlations[0]  # Normalize

            ax1.plot(
                correlations[: min(lags, len(correlations))],
                color=self.paper_style.colors[0],
                linewidth=self.paper_style.line_width,
            )
            ax1.set_title("Autocorrelation (Simplified)")
            ax1.grid(True, alpha=0.3)

            ax2.set_visible(False)

        plt.tight_layout()
        return fig


# ============================================================================
# UNIFIED ML VISUALIZER - EXTENDING EXISTING RESEARCH PLOTTER
# ============================================================================


class MLVisualizer(ResearchPlotter):
    """Dual-track ML visualizer extending existing ResearchPlotter"""

    def __init__(self, mode: str = "exploration"):
        """
        Initialize ML visualizer

        Args:
                mode: "exploration" for fast interactive plots, "publication" for high-quality plots
        """
        self.mode = mode

        if mode == "exploration":
            if HAS_HOLOVIEWS:
                self.backend = ExplorationBackend()
                self.backend_name = "holoviews"
            else:
                # Fallback to matplotlib for exploration
                super().__init__("matplotlib")
                warnings.warn(
                    "HoloViews not available. Using matplotlib for exploration."
                )
        else:  # publication mode
            self.backend = PublicationBackend()
            self.backend_name = "publication"

    def switch_mode(self, mode: str):
        """Switch between exploration and publication modes"""
        self.mode = mode
        if mode == "exploration" and HAS_HOLOVIEWS:
            self.backend = ExplorationBackend()
            self.backend_name = "holoviews"
        else:
            self.backend = PublicationBackend()
            self.backend_name = "publication"

    # ========================================================================
    # BASIC STATISTICAL CHARTS
    # ========================================================================

    def line_plot(
        self, df: pl.DataFrame, config: MLPlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Line plot - time series data, trend analysis"""
        return self.backend.create_line_plot(df, config, x_col, y_col, **kwargs)

    def scatter_plot(
        self, df: pl.DataFrame, config: MLPlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Scatter plot - correlation analysis, data distribution"""
        return self.backend.create_scatter_plot(df, config, x_col, y_col, **kwargs)

    def bar_plot(
        self, df: pl.DataFrame, config: MLPlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Bar plot - categorical data comparison"""
        return self.backend.create_bar_plot(df, config, x_col, y_col, **kwargs)

    def histogram(self, df: pl.DataFrame, config: MLPlotConfig, col: str, **kwargs):
        """Histogram - data distribution, frequency analysis"""
        return self.backend.create_histogram(df, config, col, **kwargs)

    def box_plot(
        self,
        df: pl.DataFrame,
        config: MLPlotConfig,
        y_col: str,
        x_col: Optional[str] = None,
        **kwargs,
    ):
        """Box plot - data distribution, outlier detection"""
        return self.backend.create_box_plot(df, config, y_col, x_col, **kwargs)

    def violin_plot(
        self,
        df: pl.DataFrame,
        config: MLPlotConfig,
        y_col: str,
        x_col: Optional[str] = None,
        **kwargs,
    ):
        """Violin plot - data distribution density"""
        return self.backend.create_violin_plot(df, config, y_col, x_col, **kwargs)

    # ========================================================================
    # MULTI-DIMENSIONAL DATA VISUALIZATION
    # ========================================================================

    def heatmap(self, df: pl.DataFrame, config: MLPlotConfig, **kwargs):
        """Heatmap - correlation matrix, data matrix visualization"""
        return self.backend.create_heatmap(df, config, **kwargs)

    def bubble_plot(
        self,
        df: pl.DataFrame,
        config: MLPlotConfig,
        x_col: str,
        y_col: str,
        size_col: str,
        **kwargs,
    ):
        """Bubble plot - three-dimensional data relationships"""
        return self.backend.create_bubble_plot(
            df, config, x_col, y_col, size_col, **kwargs
        )

    def radar_chart(
        self,
        df: pl.DataFrame,
        config: MLPlotConfig,
        category_col: str,
        value_cols: List[str],
        **kwargs,
    ):
        """Radar chart - multi-indicator comparison"""
        return self.backend.create_radar_chart(
            df, config, category_col, value_cols, **kwargs
        )

    def parallel_coordinates(
        self,
        df: pl.DataFrame,
        config: MLPlotConfig,
        value_cols: List[str],
        class_col: Optional[str] = None,
        **kwargs,
    ):
        """Parallel coordinates plot - high-dimensional data visualization"""
        return self.backend.create_parallel_coordinates(
            df, config, value_cols, class_col, **kwargs
        )

    # ========================================================================
    # SCIENTIFIC SPECIALIZED CHARTS
    # ========================================================================

    def error_bar_plot(
        self,
        df: pl.DataFrame,
        config: MLPlotConfig,
        x_col: str,
        y_col: str,
        error_col: str,
        **kwargs,
    ):
        """Error bar plot - experimental data uncertainty"""
        return self.backend.create_error_bar_plot(
            df, config, x_col, y_col, error_col, **kwargs
        )

    def regression_plot(
        self, df: pl.DataFrame, config: MLPlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Regression plot - fitting curves and confidence intervals"""
        return self.backend.create_regression_plot(df, config, x_col, y_col, **kwargs)

    def residual_plot(
        self,
        df: pl.DataFrame,
        config: MLPlotConfig,
        x_col: str,
        y_col: str,
        predicted_col: str,
        **kwargs,
    ):
        """Residual plot - model diagnostics"""
        return self.backend.create_residual_plot(
            df, config, x_col, y_col, predicted_col, **kwargs
        )

    def qq_plot(self, df: pl.DataFrame, config: MLPlotConfig, col: str, **kwargs):
        """Q-Q plot - distribution testing"""
        return self.backend.create_qq_plot(df, config, col, **kwargs)

    def density_plot(self, df: pl.DataFrame, config: MLPlotConfig, col: str, **kwargs):
        """Density plot - probability density distribution"""
        return self.backend.create_density_plot(df, config, col, **kwargs)

    def cdf_plot(self, df: pl.DataFrame, config: MLPlotConfig, col: str, **kwargs):
        """CDF plot - cumulative distribution function"""
        return self.backend.create_cdf_plot(df, config, col, **kwargs)

    # ========================================================================
    # MULTI-SUBPLOT COMBINATIONS
    # ========================================================================

    def subplots_grid(
        self, df: pl.DataFrame, config: MLPlotConfig, plot_configs: List[Dict], **kwargs
    ):
        """Subplot grid - multiple related chart combinations"""
        return self.backend.create_subplots_grid(df, config, plot_configs, **kwargs)

    def pair_plot(
        self, df: pl.DataFrame, config: MLPlotConfig, columns: List[str], **kwargs
    ):
        """Pair plot - pairwise variable relationships"""
        return self.backend.create_pair_plot(df, config, columns, **kwargs)

    def facet_grid(
        self,
        df: pl.DataFrame,
        config: MLPlotConfig,
        plot_type: str,
        x_col: str,
        y_col: str,
        facet_col: str,
        **kwargs,
    ):
        """Facet grid - grouped display by category"""
        return self.backend.create_facet_grid(
            df, config, plot_type, x_col, y_col, facet_col, **kwargs
        )

    # ========================================================================
    # TIME SERIES SPECIALIZED
    # ========================================================================

    def time_series_decomposition(
        self,
        df: pl.DataFrame,
        config: MLPlotConfig,
        time_col: str,
        value_col: str,
        **kwargs,
    ):
        """Time series decomposition - trend, seasonality, residual"""
        return self.backend.create_time_series_decomposition(
            df, config, time_col, value_col, **kwargs
        )

    def acf_pacf_plot(self, df: pl.DataFrame, config: MLPlotConfig, col: str, **kwargs):
        """ACF/PACF plot - time series analysis"""
        return self.backend.create_acf_pacf_plot(df, config, col, **kwargs)

    # ========================================================================
    # ML-SPECIFIC VISUALIZATION METHODS
    # ========================================================================

    def training_curves(
        self,
        df: pl.DataFrame,
        config: MLPlotConfig,
        x_col: str = "epoch",
        y_cols: List[str] = ["train_loss", "val_loss"],
        **kwargs,
    ):
        """Training curves with mode-aware rendering"""
        # Publication-quality curves
        return self.backend.create_training_curves(df, config, x_col, y_cols, **kwargs)

    def large_scatter(
        self,
        df: pl.DataFrame,
        config: MLPlotConfig,
        x_col: str,
        y_col: str,
        color_col: Optional[str] = None,
        **kwargs,
    ):
        """Large dataset scatter plot with automatic optimization"""
        if self.mode == "exploration" and HAS_HOLOVIEWS:
            return self.backend.create_scatter_plot(
                df, config, x_col, y_col, color_col=color_col, **kwargs
            )
        else:
            # For publication, sample large datasets
            if len(df) > 10000:
                df_sampled = df.sample(n=10000, seed=42)
                warnings.warn(
                    f"Sampled {len(df_sampled)} points from {len(df)} for publication plot"
                )
                return self.backend.create_scatter_plot(
                    df_sampled, config, x_col, y_col, **kwargs
                )
            else:
                return self.backend.create_scatter_plot(
                    df, config, x_col, y_col, **kwargs
                )

    def model_comparison(
        self,
        df: pl.DataFrame,
        config: MLPlotConfig,
        model_col: str,
        metric_col: str,
        error_col: Optional[str] = None,
        **kwargs,
    ):
        """Model comparison bar chart"""
        # Publication bar chart
        return self.backend.create_model_comparison(
            df, config, model_col, metric_col, error_col, **kwargs
        )

    def hyperparameter_heatmap(
        self,
        df: pl.DataFrame,
        config: MLPlotConfig,
        x_param: str,
        y_param: str,
        metric: str = "accuracy",
        **kwargs,
    ):
        """Hyperparameter optimization heatmap"""
        if self.mode == "exploration" and HAS_HOLOVIEWS:
            return self.backend.create_hyperparameter_heatmap(
                df, config, x_param, y_param, metric
            )
        else:
            # Publication heatmap using existing matplotlib backend
            config_pub = config.for_publication()
            pivot_data = df.to_pandas().pivot(
                index=y_param, columns=x_param, values=metric
            )

            self.backend._setup_figure(config_pub)

            if HAS_SEABORN:
                sns.heatmap(
                    pivot_data,
                    annot=True,
                    fmt=".3f",
                    cmap="viridis",
                    ax=self.backend.ax,
                    cbar_kws={"label": metric},
                )
            else:
                im = self.backend.ax.imshow(
                    pivot_data.values, cmap="viridis", aspect="auto"
                )
                self.backend.fig.colorbar(im, ax=self.backend.ax, label=metric)

            self.backend.ax.set_xlabel(x_param.replace("_", " ").title())
            self.backend.ax.set_ylabel(y_param.replace("_", " ").title())
            self.backend.ax.set_title(config.title)

            return self.backend.fig

    # ========================================================================
    # WORKFLOW METHODS
    # ========================================================================

    def exploration_to_publication(self, plot_func, *args, **kwargs):
        """Convert exploration plot to publication quality"""
        # Save current mode
        original_mode = self.mode

        # Switch to publication mode
        self.switch_mode("publication")

        # Create publication plot
        try:
            fig = plot_func(*args, **kwargs)
            return fig
        finally:
            # Restore original mode
            self.switch_mode(original_mode)

    def save_publication_figure(
        self, fig, filename: Union[str, Path], config: MLPlotConfig
    ):
        """Save figure with publication settings"""
        pub_config = config.for_publication()
        filename = str(filename)

        # Ensure proper file extension
        if not filename.endswith((".pdf", ".eps", ".tiff", ".png")):
            filename += f".{pub_config.export_format}"

        if hasattr(fig, "savefig"):  # matplotlib figure
            fig.savefig(
                filename,
                format=pub_config.export_format,
                dpi=pub_config.export_dpi,
                bbox_inches="tight",
                pad_inches=0.05,
            )
        else:  # HoloViews figure
            hv.save(fig, filename)
