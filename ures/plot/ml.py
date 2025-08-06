"""
Machine Learning Visualization Framework
Built on ures.plot.backend infrastructure with dual-track approach:
- Exploration: HoloViews + Datashader (Speed + Interactivity)
- Publication: Matplotlib + Seaborn (Quality + Control)
"""

import polars as pl
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field

# Import from ures.plot.backend
from ures.plot.backend import (
    PlotConfig,
    ColorConfig,
    ColorScheme,
    PlotBackend,
    validate_dataframe,
    ResearchBackend,
    HAS_SEABORN,
    check_optional_dependency,
)

# Optional ML-specific dependencies
HAS_HOLOVIEWS = check_optional_dependency("holoviews")
HAS_DATASHADER = check_optional_dependency("datashader")
HAS_SKLEARN = check_optional_dependency("sklearn")

if HAS_HOLOVIEWS:
    import holoviews as hv
    import panel as pn
    from holoviews.operation.datashader import datashade, shade, dynspread

    hv.extension("bokeh", "matplotlib")

if HAS_DATASHADER:
    import datashader as ds

if HAS_SEABORN:
    import seaborn as sns

if HAS_SKLEARN:
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA


# ============================================================================
# ENHANCED ML PLOT CONFIGURATION
# ============================================================================


@dataclass
class MLPlotConfig(PlotConfig):
    """ML-specific plot configuration extending base PlotConfig"""

    # ML experiment settings
    experiment_name: str = "ml_experiment"
    model_name: str = "model"
    metric_name: str = "accuracy"

    # Visualization mode
    mode: str = "exploration"  # "exploration" or "publication"

    # Large data handling
    enable_datashading: bool = True
    datashade_threshold: int = 50000
    sample_large_data: bool = True
    max_sample_size: int = 10000

    # ML-specific color schemes
    ml_color_scheme: str = "scientific"  # scientific, colorblind, viridis

    # Publication settings override
    publication_dpi: int = 300
    publication_format: str = "pdf"
    publication_font_family: str = "Arial"

    # Interactive features
    enable_hover_tools: bool = True
    enable_selection_tools: bool = True
    enable_zoom_tools: bool = True

    def for_publication(self) -> "MLPlotConfig":
        """Create publication-ready configuration"""
        pub_config = MLPlotConfig(
            # Copy all base attributes
            **{k: v for k, v in self.__dict__.items()},
            # Override for publication
            mode="publication",
            style="seaborn-v0_8-paper" if HAS_SEABORN else "whitegrid",
            export_dpi=self.publication_dpi,
            export_format=self.publication_format,
            smooth=False,
            enable_datashading=False,  # Use raw data for publication
            color_config=ColorConfig(scheme=ColorScheme.COLORBLIND_FRIENDLY),
            font_config={
                "title_size": 10,
                "label_size": 8,
                "tick_size": 7,
                "legend_size": 7,
            },
            tight_layout=True,
        )
        return pub_config


# ============================================================================
# HOLOVIEWS EXPLORATION BACKEND
# ============================================================================


class HoloViewsMLBackend(PlotBackend):
    """HoloViews backend for fast ML exploration with datashading"""

    def __init__(self):
        if not HAS_HOLOVIEWS:
            raise ImportError(
                "HoloViews required. Install: pip install holoviews bokeh"
            )

        # Configure HoloViews defaults
        hv.opts.defaults(
            hv.opts.Curve(tools=["hover", "box_zoom"], width=800, height=400),
            hv.opts.Scatter(
                tools=["hover", "box_select", "lasso_select"], width=800, height=600
            ),
            hv.opts.Bars(tools=["hover"], width=600, height=400),
            hv.opts.Image(tools=["hover"], width=600, height=400),
        )

    def _setup_hover_tools(self, element, config: MLPlotConfig):
        """Setup interactive hover tools"""
        if not config.enable_hover_tools:
            return element

        hover_cols = []
        if hasattr(element, "data") and hasattr(element.data, "columns"):
            hover_cols = list(element.data.columns)[:5]  # Limit hover columns

        return element.opts(tools=["hover"])

    def _handle_large_data(
        self,
        df: pl.DataFrame,
        config: MLPlotConfig,
        x_col: str,
        y_col: str,
        color_col: Optional[str] = None,
    ):
        """Handle large datasets with datashading or sampling"""

        if len(df) <= config.datashade_threshold:
            return df, False  # No processing needed

        if config.enable_datashading and HAS_DATASHADER:
            return df, True  # Use datashading

        if config.sample_large_data:
            # Sample data for performance
            sample_size = min(config.max_sample_size, len(df))
            df_sampled = df.sample(n=sample_size, seed=42)
            warnings.warn(
                f"Sampled {sample_size} points from {len(df)} for visualization"
            )
            return df_sampled, False

        return df, False

    # Core plot implementations
    def create_line_plot(
        self, df: pl.DataFrame, config: MLPlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """High-performance line plot with HoloViews"""
        validate_dataframe(df, required_cols=[x_col, y_col])

        # Handle grouping for multiple lines
        group_col = kwargs.get("group_col", None)

        if group_col:
            # Multiple line series
            curves = []
            colors = config.color_config.get_colors()

            for i, group in enumerate(df[group_col].unique()):
                group_data = df.filter(pl.col(group_col) == group)
                curve = hv.Curve(group_data.to_pandas(), x_col, y_col, label=str(group))
                curve = curve.opts(color=colors[i % len(colors)], line_width=2)
                curves.append(curve)

            overlay = hv.Overlay(curves)
            return overlay.opts(
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
                legend_position="right",
                width=800,
                height=400,
            )
        else:
            # Single line
            curve = hv.Curve(df.to_pandas(), x_col, y_col)
            return curve.opts(
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
                color=config.color_config.get_color(0),
                line_width=2,
                tools=(
                    ["hover", "box_zoom", "reset"] if config.enable_hover_tools else []
                ),
            )

    def create_scatter_plot(
        self, df: pl.DataFrame, config: MLPlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """High-performance scatter plot with auto-datashading"""
        validate_dataframe(df, required_cols=[x_col, y_col])

        color_col = kwargs.get("color_col", None)
        size_col = kwargs.get("size_col", None)

        # Handle large datasets
        df_viz, use_datashade = self._handle_large_data(
            df, config, x_col, y_col, color_col
        )

        # Create points
        dims = [x_col, y_col]
        if color_col:
            dims.append(color_col)
        if size_col:
            dims.append(size_col)

        points = hv.Points(df_viz.to_pandas(), dims)

        # Apply datashading for large datasets
        if use_datashade and HAS_DATASHADER:
            if color_col:
                return datashade(
                    points, color_key=color_col, cmap="viridis", width=800, height=600
                )
            else:
                return datashade(points, cmap="blues", width=800, height=600)

        # Standard scatter plot
        opts = {
            "title": config.title,
            "xlabel": (
                f"{config.xlabel} ({config.x_unit})" if config.x_unit else config.xlabel
            ),
            "ylabel": (
                f"{config.ylabel} ({config.y_unit})" if config.y_unit else config.ylabel
            ),
            "size": 6,
            "alpha": 0.7,
            "width": 800,
            "height": 600,
        }

        if config.enable_hover_tools:
            opts["tools"] = ["hover", "box_select", "lasso_select"]

        if color_col:
            opts["color"] = color_col
            opts["cmap"] = "viridis"
        else:
            opts["color"] = config.color_config.get_color(0)

        if size_col:
            opts["size"] = size_col

        return points.opts(**opts)

    def create_bar_plot(
        self, df: pl.DataFrame, config: MLPlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Interactive bar plot"""
        validate_dataframe(df, required_cols=[x_col, y_col])

        bars = hv.Bars(df.to_pandas(), x_col, y_col)

        opts = {
            "title": config.title,
            "xlabel": config.xlabel or x_col,
            "ylabel": config.ylabel or y_col,
            "color": config.color_config.get_color(0),
            "width": 600,
            "height": 400,
        }

        if config.enable_hover_tools:
            opts["tools"] = ["hover"]

        return bars.opts(**opts)

    def create_histogram(
        self, df: pl.DataFrame, config: MLPlotConfig, col: str, **kwargs
    ):
        """Interactive histogram with optional density overlay"""
        validate_dataframe(df, required_cols=[col])

        bins = kwargs.get("bins", 30)

        hist = hv.Histogram(np.histogram(df[col].to_numpy(), bins=bins))

        opts = {
            "title": config.title,
            "xlabel": config.xlabel or col,
            "ylabel": "Frequency",
            "color": config.color_config.get_color(0),
            "alpha": 0.7,
            "width": 600,
            "height": 400,
        }

        if config.enable_hover_tools:
            opts["tools"] = ["hover"]

        return hist.opts(**opts)

    # ML-specific implementations
    def create_training_curves(
        self,
        df: pl.DataFrame,
        config: MLPlotConfig,
        x_col: str = "epoch",
        y_cols: List[str] = ["train_loss", "val_loss"],
        **kwargs,
    ):
        """Interactive training curves visualization"""
        validate_dataframe(df, required_cols=[x_col] + y_cols)

        curves = []
        colors = config.color_config.get_colors()

        for i, y_col in enumerate(y_cols):
            curve = hv.Curve(df.to_pandas(), x_col, y_col, label=y_col)
            curve = curve.opts(
                color=colors[i % len(colors)],
                line_width=2,
                tools=["hover"] if config.enable_hover_tools else [],
            )
            curves.append(curve)

        overlay = hv.Overlay(curves)
        return overlay.opts(
            title=config.title or "Training Curves",
            xlabel=config.xlabel or x_col.replace("_", " ").title(),
            ylabel=config.ylabel or "Loss/Metric",
            legend_position="right",
            width=800,
            height=400,
        )

    def create_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        config: MLPlotConfig,
        class_names: Optional[List[str]] = None,
    ):
        """Interactive confusion matrix heatmap"""
        if not HAS_SKLEARN:
            raise ImportError("sklearn required for confusion matrix")

        cm = confusion_matrix(y_true, y_pred)

        if class_names is None:
            class_names = [f"Class_{i}" for i in range(len(cm))]

        # Convert to DataFrame for HoloViews
        cm_data = []
        for i, true_label in enumerate(class_names):
            for j, pred_label in enumerate(class_names):
                cm_data.append(
                    {
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "count": cm[i, j],
                    }
                )

        cm_df = pl.DataFrame(cm_data)

        heatmap = hv.HeatMap(cm_df.to_pandas(), ["pred_label", "true_label"], "count")

        return heatmap.opts(
            title="Confusion Matrix",
            xlabel="Predicted Label",
            ylabel="True Label",
            width=500,
            height=500,
            cmap="blues",
            tools=["hover"] if config.enable_hover_tools else [],
            colorbar=True,
        )

    def create_feature_importance(
        self, importance_data: Dict[str, float], config: MLPlotConfig, top_n: int = 20
    ):
        """Interactive feature importance plot"""
        # Convert to DataFrame
        features_df = (
            pl.DataFrame(
                {
                    "feature": list(importance_data.keys()),
                    "importance": list(importance_data.values()),
                }
            )
            .sort("importance", descending=True)
            .head(top_n)
        )

        bars = hv.Bars(features_df.to_pandas(), "feature", "importance")

        return bars.opts(
            title="Feature Importance",
            xlabel="Features",
            ylabel="Importance",
            width=800,
            height=400,
            tools=["hover"] if config.enable_hover_tools else [],
            xrotation=45,
        )

    # Required abstract methods from PlotBackend
    def apply_annotations(self, fig, config: PlotConfig):
        """Apply annotations (limited support in HoloViews)"""
        # HoloViews handles annotations differently, minimal implementation
        pass

    def export_plot(self, fig, filepath: str, config: PlotConfig):
        """Export HoloViews plot"""
        try:
            hv.save(fig, filepath)
        except Exception as e:
            warnings.warn(f"Failed to export HoloViews plot: {e}")

    def apply_styling(self, fig, config: PlotConfig):
        """Apply styling (handled through opts in HoloViews)"""
        pass


# ============================================================================
# MATPLOTLIB PUBLICATION BACKEND
# ============================================================================


class MatplotlibMLBackend(ResearchBackend):
    """Extended Matplotlib backend for ML publication plots"""

    def __init__(self):
        super().__init__()
        self._setup_ml_style()

    def _setup_ml_style(self):
        """Setup ML-specific matplotlib styling"""
        import matplotlib.pyplot as plt

        # Publication-ready settings
        plt.rcParams.update(
            {
                "font.family": "Arial",
                "font.size": 8,
                "axes.linewidth": 0.5,
                "lines.linewidth": 1.0,
                "patch.linewidth": 0.5,
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.05,
                "figure.constrained_layout.use": True,
            }
        )

        if HAS_SEABORN:
            sns.set_palette("colorblind")
            sns.set_context("paper")

    def create_training_curves(
        self,
        df: pl.DataFrame,
        config: MLPlotConfig,
        x_col: str = "epoch",
        y_cols: List[str] = ["train_loss", "val_loss"],
        **kwargs,
    ):
        """Publication-quality training curves"""
        validate_dataframe(df, required_cols=[x_col] + y_cols)
        self._setup_figure(config)

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        for i, y_col in enumerate(y_cols):
            if HAS_SEABORN:
                sns.lineplot(
                    data=df.to_pandas(),
                    x=x_col,
                    y=y_col,
                    ax=self.ax,
                    color=colors[i % len(colors)],
                    linewidth=1.5,
                    label=y_col.replace("_", " ").title(),
                )
            else:
                self.ax.plot(
                    df[x_col],
                    df[y_col],
                    color=colors[i % len(colors)],
                    linewidth=1.5,
                    label=y_col.replace("_", " ").title(),
                )

        self.ax.set_xlabel(config.xlabel or x_col.replace("_", " ").title())
        self.ax.set_ylabel(config.ylabel or "Value")
        self.ax.set_title(config.title or "Training Progress")
        self.ax.legend(fontsize=config.font_config.get("legend_size", 8))
        self.ax.grid(True, alpha=0.3)

        return self.fig

    def create_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        config: MLPlotConfig,
        class_names: Optional[List[str]] = None,
    ):
        """Publication-quality confusion matrix"""
        if not HAS_SKLEARN:
            raise ImportError("sklearn required for confusion matrix")

        self._setup_figure(config)

        cm = confusion_matrix(y_true, y_pred)

        if HAS_SEABORN:
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=self.ax,
                xticklabels=class_names,
                yticklabels=class_names,
            )
        else:
            import matplotlib.pyplot as plt

            im = self.ax.imshow(cm, cmap="Blues")
            self.fig.colorbar(im, ax=self.ax)

            # Add text annotations
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    self.ax.text(j, i, str(cm[i, j]), ha="center", va="center")

        self.ax.set_xlabel("Predicted Label")
        self.ax.set_ylabel("True Label")
        self.ax.set_title("Confusion Matrix")

        return self.fig

    def create_feature_importance(
        self, importance_data: Dict[str, float], config: MLPlotConfig, top_n: int = 20
    ):
        """Publication-quality feature importance plot"""
        self._setup_figure(config)

        # Sort and limit features
        sorted_features = sorted(
            importance_data.items(), key=lambda x: x[1], reverse=True
        )[:top_n]

        features, importances = zip(*sorted_features)

        if HAS_SEABORN:
            feature_df = pl.DataFrame({"feature": features, "importance": importances})

            sns.barplot(
                data=feature_df.to_pandas(),
                x="importance",
                y="feature",
                ax=self.ax,
                color=config.color_config.get_color(0),
            )
        else:
            y_pos = np.arange(len(features))
            self.ax.barh(y_pos, importances, color=config.color_config.get_color(0))
            self.ax.set_yticks(y_pos)
            self.ax.set_yticklabels(features)

        self.ax.set_xlabel("Importance")
        self.ax.set_ylabel("Features")
        self.ax.set_title("Feature Importance")
        self.ax.grid(True, alpha=0.3)

        return self.fig

    def create_roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        config: MLPlotConfig,
        pos_label: int = 1,
    ):
        """Publication-quality ROC curve"""
        if not HAS_SKLEARN:
            raise ImportError("sklearn required for ROC curve")

        from sklearn.metrics import roc_curve, auc

        self._setup_figure(config)

        fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)

        self.ax.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=1.5,
            label=f"ROC Curve (AUC = {roc_auc:.2f})",
        )
        self.ax.plot(
            [0, 1],
            [0, 1],
            color="navy",
            lw=1.5,
            linestyle="--",
            label="Random Classifier",
        )

        self.ax.set_xlim([0.0, 1.0])
        self.ax.set_ylim([0.0, 1.05])
        self.ax.set_xlabel("False Positive Rate")
        self.ax.set_ylabel("True Positive Rate")
        self.ax.set_title("Receiver Operating Characteristic (ROC) Curve")
        self.ax.legend(loc="lower right")
        self.ax.grid(True, alpha=0.3)

        return self.fig
