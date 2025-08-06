"""
Research Plotting Utilities - Fixed Dependencies Version
Supports matplotlib and plotly dual backends with polars DataFrame
Fixed for environments without scipy and other optional dependencies
"""

from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import warnings

# Optional imports with fallbacks
try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    warnings.warn("Seaborn not available. Some styling options will be limited.")

try:
    import plotly.graph_objects as go
    import plotly.express as px

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    warnings.warn("Plotly not available. Plotly backend will not work.")

try:
    from scipy import stats
    from scipy.interpolate import interp1d

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn(
        "Scipy not available. Advanced smoothing and statistics will be limited."
    )

from abc import ABC, abstractmethod


# 1. Annotation System Design
@dataclass
class Annotation:
    """Image annotation configuration class"""

    text: str
    x: float
    y: float
    annotation_type: str = "text"  # text, arrow, box, circle

    # Text style parameters
    fontsize: int = 10
    fontweight: str = "normal"
    color: str = "black"

    # Arrow annotation parameters
    arrow_props: Optional[Dict] = None

    # Box annotation parameters
    bbox_props: Optional[Dict] = None

    # Relative/absolute coordinates
    xycoords: str = "data"  # data, axes fraction, figure fraction

    # Plotly specific parameters
    showarrow: bool = True
    arrowhead: int = 2


@dataclass
class AnnotationCollection:
    """Annotation collection management"""

    annotations: List[Annotation] = field(default_factory=list)

    def add_annotation(self, annotation: Annotation):
        """Add annotation to collection"""
        self.annotations.append(annotation)

    def add_text(self, text: str, x: float, y: float, **kwargs):
        """Quick add text annotation"""
        ann = Annotation(text=text, x=x, y=y, **kwargs)
        self.add_annotation(ann)

    def add_arrow(
        self, text: str, x: float, y: float, arrow_x: float, arrow_y: float, **kwargs
    ):
        """Quick add arrow annotation"""
        arrow_props = kwargs.pop(
            "arrow_props",
            {"arrowstyle": "->", "connectionstyle": "arc3,rad=0", "color": "red"},
        )
        ann = Annotation(
            text=text,
            x=x,
            y=y,
            annotation_type="arrow",
            arrow_props=arrow_props,
            **kwargs,
        )
        self.add_annotation(ann)


# 2. Color Scheme Management
class ColorScheme(Enum):
    """Predefined color schemes"""

    SCIENTIFIC = "scientific"
    NATURE = "nature"
    SCIENCE = "science"
    COLORBLIND_FRIENDLY = "colorblind"
    CUSTOM = "custom"


@dataclass
class ColorConfig:
    """Color configuration management"""

    scheme: ColorScheme = ColorScheme.SCIENTIFIC
    custom_colors: Optional[List[str]] = None

    # Predefined color schemes
    _color_schemes = {
        ColorScheme.SCIENTIFIC: ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
        ColorScheme.NATURE: ["#0173B2", "#DE8F05", "#029E73", "#CC78BC", "#CA9161"],
        ColorScheme.SCIENCE: ["#3182bd", "#e6550d", "#31a354", "#756bb1", "#636363"],
        ColorScheme.COLORBLIND_FRIENDLY: [
            "#377eb8",
            "#ff7f00",
            "#4daf4a",
            "#f781bf",
            "#a65628",
        ],
    }

    def get_colors(self) -> List[str]:
        """Get color list based on current scheme"""
        if self.scheme == ColorScheme.CUSTOM and self.custom_colors:
            return self.custom_colors
        return self._color_schemes.get(
            self.scheme, self._color_schemes[ColorScheme.SCIENTIFIC]
        )


# 3. Main Configuration Class
@dataclass
class PlotConfig:
    """Main plot configuration class"""

    # Basic settings
    figsize: Tuple[Union[int, float], Union[int, float]] = (10, 6)
    dpi: int = 100
    style: str = "whitegrid"

    # Axis settings
    xlabel: str = ""
    ylabel: str = ""
    title: str = ""
    grid: bool = True

    # Unit settings
    x_unit: str = ""
    y_unit: str = ""
    unit_position: str = "label"  # label, tick, both

    # Color configuration
    color_config: ColorConfig = field(default_factory=ColorConfig)

    # Smoothing settings
    smooth: bool = False
    smooth_method: str = "spline"  # spline, lowess, savgol
    smooth_factor: float = 0.3

    # Legend settings
    legend: bool = True
    legend_position: str = "best"

    # Annotation settings
    annotations: AnnotationCollection = field(default_factory=AnnotationCollection)

    # Export settings
    export_format: str = "png"
    export_dpi: int = 300
    export_transparent: bool = False

    # Font settings
    font_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "title_size": 14,
            "label_size": 12,
            "tick_size": 10,
            "legend_size": 10,
        }
    )


# 4. Backend Interface
class PlotBackend(ABC):
    """Abstract plotting backend interface"""

    @abstractmethod
    def create_line_plot(self, df: pl.DataFrame, config: PlotConfig, **kwargs):
        """Create line plot"""
        pass

    @abstractmethod
    def create_scatter_plot(self, df: pl.DataFrame, config: PlotConfig, **kwargs):
        """Create scatter plot"""
        pass

    @abstractmethod
    def apply_annotations(self, fig, config: PlotConfig):
        """Apply annotations to figure"""
        pass

    @abstractmethod
    def export_plot(self, fig, filepath: str, config: PlotConfig):
        """Export plot to file"""
        pass


# 5. Matplotlib Backend Implementation
class MatplotlibBackend(PlotBackend):
    """Matplotlib backend implementation"""

    def __init__(self):
        self.fig = None
        self.ax = None

    def _setup_figure(self, config: PlotConfig):
        """Setup basic figure parameters"""
        # Use available styles with fallbacks
        available_styles = plt.style.available

        if config.style == "whitegrid":
            if "seaborn-v0_8-whitegrid" in available_styles:
                plt.style.use("seaborn-v0_8-whitegrid")
            elif "seaborn-whitegrid" in available_styles:
                plt.style.use("seaborn-whitegrid")
            elif HAS_SEABORN:
                sns.set_style("whitegrid")
            else:
                # Fallback to default
                plt.style.use("default")
        elif config.style in available_styles:
            plt.style.use(config.style)
        else:
            plt.style.use("default")

        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

        # Set labels and title
        self._set_labels(config)

    def _set_labels(self, config: PlotConfig):
        """Set axis labels and units"""
        xlabel = config.xlabel
        ylabel = config.ylabel

        if config.x_unit and config.unit_position in ["label", "both"]:
            xlabel = f"{xlabel} ({config.x_unit})" if xlabel else f"({config.x_unit})"
        if config.y_unit and config.unit_position in ["label", "both"]:
            ylabel = f"{ylabel} ({config.y_unit})" if ylabel else f"({config.y_unit})"

        self.ax.set_xlabel(xlabel, fontsize=config.font_config["label_size"])
        self.ax.set_ylabel(ylabel, fontsize=config.font_config["label_size"])
        self.ax.set_title(config.title, fontsize=config.font_config["title_size"])

    def _apply_smoothing(self, x_data, y_data, config: PlotConfig):
        """Apply data smoothing with fallbacks"""
        if not config.smooth:
            return x_data, y_data

        if not HAS_SCIPY:
            warnings.warn("Scipy not available. Smoothing disabled.")
            return x_data, y_data

        # Implementation of different smoothing algorithms
        try:
            if config.smooth_method == "spline":
                f = interp1d(x_data, y_data, kind="cubic")
                x_smooth = np.linspace(x_data.min(), x_data.max(), len(x_data) * 3)
                y_smooth = f(x_smooth)
                return x_smooth, y_smooth
            else:
                # Simple moving average fallback
                window = max(3, len(x_data) // 20)
                y_smooth = np.convolve(y_data, np.ones(window) / window, mode="same")
                return x_data, y_smooth
        except Exception as e:
            warnings.warn(f"Smoothing failed: {e}. Using original data.")
            return x_data, y_data

    def create_line_plot(
        self, df: pl.DataFrame, config: PlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Create line plot"""
        self._setup_figure(config)

        colors = config.color_config.get_colors()
        x_data = df[x_col].to_numpy()
        y_data = df[y_col].to_numpy()

        # Apply smoothing
        x_smooth, y_smooth = self._apply_smoothing(x_data, y_data, config)

        self.ax.plot(
            x_smooth,
            y_smooth,
            color=colors[0],
            linewidth=kwargs.get("linewidth", 2),
            **{k: v for k, v in kwargs.items() if k not in ["linewidth"]},
        )

        self.apply_annotations(self.fig, config)
        return self.fig

    def create_scatter_plot(
        self, df: pl.DataFrame, config: PlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Create scatter plot"""
        self._setup_figure(config)

        colors = config.color_config.get_colors()

        self.ax.scatter(
            df[x_col],
            df[y_col],
            c=colors[0],
            s=kwargs.get("s", 50),
            alpha=kwargs.get("alpha", 0.7),
            **{k: v for k, v in kwargs.items() if k not in ["s", "alpha"]},
        )

        self.apply_annotations(self.fig, config)
        return self.fig

    def apply_annotations(self, fig, config: PlotConfig):
        """Apply annotations to matplotlib figure"""
        for ann in config.annotations.annotations:
            try:
                if ann.annotation_type == "text":
                    self.ax.annotate(
                        ann.text,
                        (ann.x, ann.y),
                        fontsize=ann.fontsize,
                        color=ann.color,
                        xycoords=ann.xycoords,
                    )
                elif ann.annotation_type == "arrow":
                    self.ax.annotate(
                        ann.text,
                        (ann.x, ann.y),
                        arrowprops=ann.arrow_props,
                        fontsize=ann.fontsize,
                        color=ann.color,
                    )
            except Exception as e:
                warnings.warn(f"Failed to add annotation: {e}")

    def export_plot(self, fig, filepath: str, config: PlotConfig):
        """Export matplotlib figure"""
        try:
            fig.savefig(
                filepath,
                format=config.export_format,
                dpi=config.export_dpi,
                transparent=config.export_transparent,
                bbox_inches="tight",
            )
            plt.close(fig)
        except Exception as e:
            warnings.warn(f"Failed to export plot: {e}")


# 6. Plotly Backend Implementation
class PlotlyBackend(PlotBackend):
    """Plotly backend implementation"""

    def __init__(self):
        if not HAS_PLOTLY:
            raise ImportError("Plotly not available. Install with: pip install plotly")

    def create_line_plot(
        self, df: pl.DataFrame, config: PlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Create Plotly line plot"""
        colors = config.color_config.get_colors()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode="lines",
                line=dict(color=colors[0], width=kwargs.get("linewidth", 2)),
                name=kwargs.get("name", "Line"),
            )
        )

        self._setup_plotly_layout(fig, config)
        self.apply_annotations(fig, config)
        return fig

    def create_scatter_plot(
        self, df: pl.DataFrame, config: PlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Create Plotly scatter plot"""
        colors = config.color_config.get_colors()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode="markers",
                marker=dict(color=colors[0], size=kwargs.get("size", 8)),
                name=kwargs.get("name", "Scatter"),
            )
        )

        self._setup_plotly_layout(fig, config)
        self.apply_annotations(fig, config)
        return fig

    def _setup_plotly_layout(self, fig, config: PlotConfig):
        """Setup Plotly layout"""
        xlabel = config.xlabel
        ylabel = config.ylabel

        if config.x_unit:
            xlabel = f"{xlabel} ({config.x_unit})" if xlabel else f"({config.x_unit})"
        if config.y_unit:
            ylabel = f"{ylabel} ({config.y_unit})" if ylabel else f"({config.y_unit})"

        fig.update_layout(
            title=config.title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            width=config.figsize[0] * 100,
            height=config.figsize[1] * 100,
            font=dict(size=config.font_config["label_size"]),
        )

    def apply_annotations(self, fig, config: PlotConfig):
        """Apply Plotly annotations"""
        for ann in config.annotations.annotations:
            try:
                fig.add_annotation(
                    text=ann.text,
                    x=ann.x,
                    y=ann.y,
                    showarrow=ann.showarrow,
                    arrowhead=ann.arrowhead,
                    font=dict(size=ann.fontsize, color=ann.color),
                )
            except Exception as e:
                warnings.warn(f"Failed to add annotation: {e}")

    def export_plot(self, fig, filepath: str, config: PlotConfig):
        """Export Plotly figure"""
        try:
            if config.export_format.lower() == "html":
                fig.write_html(filepath)
            else:
                # This requires kaleido package
                fig.write_image(filepath, format=config.export_format)
        except Exception as e:
            warnings.warn(f"Failed to export plot: {e}. Try: pip install kaleido")


# 7. Main Plotting Class
class ResearchPlotter:
    """Main research plotting class"""

    def __init__(self, backend: str = "matplotlib"):
        self.backend_name = backend
        self.backend = self._create_backend(backend)

    def _create_backend(self, backend_name: str) -> PlotBackend:
        """Create backend instance"""
        if backend_name.lower() == "matplotlib":
            return MatplotlibBackend()
        elif backend_name.lower() == "plotly":
            return PlotlyBackend()
        else:
            raise ValueError(f"Unsupported backend: {backend_name}")

    def switch_backend(self, backend: str):
        """Switch plotting backend"""
        self.backend = self._create_backend(backend)
        self.backend_name = backend

    def line_plot(
        self, df: pl.DataFrame, config: PlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Create line plot"""
        return self.backend.create_line_plot(df, config, x_col, y_col, **kwargs)

    def scatter_plot(
        self, df: pl.DataFrame, config: PlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Create scatter plot"""
        return self.backend.create_scatter_plot(df, config, x_col, y_col, **kwargs)

    def export(self, fig, filepath: str, config: PlotConfig):
        """Export figure"""
        self.backend.export_plot(fig, filepath, config)


# 8. Usage Example
def example_usage():
    """Usage example"""
    # Create sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, 100)

    df = pl.DataFrame({"x": x, "y": y})

    # Configure plot
    config = PlotConfig(
        title="Research Data Visualization Example",
        xlabel="Time",
        ylabel="Signal Strength",
        x_unit="seconds",
        y_unit="mV",
        smooth=True,
        figsize=(12, 8),
    )

    # Add annotations
    config.annotations.add_text("Peak", 5, 1, fontsize=12, color="red")
    config.annotations.add_arrow("Anomaly", 7, -0.5, 7.2, -0.8, color="blue")

    # Create plotter
    plotter = ResearchPlotter("matplotlib")

    # Create plot
    fig = plotter.line_plot(df, config, "x", "y")

    # Export plot
    plotter.export(fig, "research_plot.png", config)

    # Switch to Plotly backend if available
    if HAS_PLOTLY:
        try:
            plotter.switch_backend("plotly")
            fig_plotly = plotter.line_plot(df, config, "x", "y")
            plotter.export(fig_plotly, "research_plot.html", config)
        except Exception as e:
            print(f"Plotly backend failed: {e}")


# Dependency checker
def check_dependencies():
    """Check available dependencies"""
    print("Checking available dependencies:")
    print(f"- Matplotlib: ✓ (required)")
    print(f"- Polars: ✓ (required)")
    print(f"- NumPy: ✓ (required)")
    print(f"- Seaborn: {'✓' if HAS_SEABORN else '✗ (optional - enhanced styling)'}")
    print(f"- Plotly: {'✓' if HAS_PLOTLY else '✗ (optional - interactive plots)'}")
    print(f"- SciPy: {'✓' if HAS_SCIPY else '✗ (optional - advanced smoothing)'}")

    if not HAS_SEABORN:
        print("\nTo enable enhanced styling: pip install seaborn")
    if not HAS_PLOTLY:
        print("To enable interactive plots: pip install plotly")
    if not HAS_SCIPY:
        print("To enable advanced smoothing: pip install scipy")


if __name__ == "__main__":
    check_dependencies()
    example_usage()
