from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import polars as pl
import warnings
from abc import ABC, abstractmethod
from .deps import HAS_STATSMODELS

if HAS_STATSMODELS:
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        from statsmodels.stats.diagnostic import acf, pacf
    except ImportError:
        HAS_STATSMODELS = False


# ============================================================================
# ENHANCED CONFIGURATION CLASSES
# ============================================================================


@dataclass
class Annotation:
    """Enhanced annotation configuration class"""

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

    # Enhanced features
    rotation: float = 0.0
    ha: str = "center"  # horizontal alignment
    va: str = "center"  # vertical alignment
    alpha: float = 1.0


@dataclass
class AnnotationCollection:
    """Enhanced annotation collection management"""

    annotations: List[Annotation] = field(default_factory=list)

    def add_annotation(self, annotation: Annotation):
        """Add annotation to collection"""
        self.annotations.append(annotation)

    def add_text(self, text: str, x: float, y: float, **kwargs):
        """Quick add text annotation"""
        ann = Annotation(text=text, x=x, y=y, **kwargs)
        self.add_annotation(ann)

    def add_arrow(
        self,
        text: str,
        x: float,
        y: float,
        xy_text: Tuple[float, float] = None,
        **kwargs,
    ):
        """Quick add arrow annotation"""
        arrow_props = kwargs.pop(
            "arrow_props", {"arrowstyle": "->", "color": "red", "lw": 1.5}
        )

        # If no text position specified, offset from point
        if xy_text is None:
            xy_text = (x + 0.1, y + 0.1)

        ann = Annotation(
            text=text,
            x=xy_text[0],
            y=xy_text[1],
            annotation_type="arrow",
            arrow_props=arrow_props,
            xycoords="data",
            **kwargs,
        )
        # Add the point to annotate
        ann.xytext = (x, y)
        self.add_annotation(ann)

    def add_box(self, text: str, x: float, y: float, **kwargs):
        """Quick add box annotation"""
        bbox_props = kwargs.pop(
            "bbox_props",
            {"boxstyle": "round,pad=0.3", "facecolor": "yellow", "alpha": 0.7},
        )
        ann = Annotation(
            text=text, x=x, y=y, annotation_type="box", bbox_props=bbox_props, **kwargs
        )
        self.add_annotation(ann)

    def clear(self):
        """Clear all annotations"""
        self.annotations.clear()

    def remove_last(self):
        """Remove last added annotation"""
        if self.annotations:
            self.annotations.pop()


class ColorScheme(Enum):
    """Enhanced predefined color schemes"""

    SCIENTIFIC = "scientific"
    NATURE = "nature"
    SCIENCE = "science"
    COLORBLIND_FRIENDLY = "colorblind"
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    CUSTOM = "custom"


@dataclass
class ColorConfig:
    """Enhanced color configuration management"""

    scheme: ColorScheme = ColorScheme.SCIENTIFIC
    custom_colors: Optional[List[str]] = None

    # Enhanced predefined color schemes
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
        ColorScheme.VIRIDIS: ["#440154", "#31688e", "#35b779", "#fde725"],
        ColorScheme.PLASMA: ["#0d0887", "#7e03a8", "#cc4778", "#f0f921"],
    }

    def get_colors(self) -> List[str]:
        """Get color list based on current scheme"""
        if self.scheme == ColorScheme.CUSTOM and self.custom_colors:
            return self.custom_colors
        return self._color_schemes.get(
            self.scheme, self._color_schemes[ColorScheme.SCIENTIFIC]
        )

    def get_color(self, index: int) -> str:
        """Get single color by index with cycling"""
        colors = self.get_colors()
        return colors[index % len(colors)]


@dataclass
class PlotConfig:
    """Enhanced main plot configuration class"""

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

    # Enhanced features
    tight_layout: bool = True
    show_spines: bool = True
    spine_alpha: float = 0.8


# ============================================================================
# ENHANCED BACKEND INTERFACE
# ============================================================================


class PlotBackend(ABC):
    """Enhanced abstract plotting backend interface"""

    # Core plots (must implement)
    @abstractmethod
    def create_line_plot(
        self, df: pl.DataFrame, config: PlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Create line plot"""
        pass

    @abstractmethod
    def create_scatter_plot(
        self, df: pl.DataFrame, config: PlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Create scatter plot"""
        pass

    @abstractmethod
    def create_bar_plot(
        self, df: pl.DataFrame, config: PlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Create bar plot"""
        pass

    @abstractmethod
    def create_histogram(
        self, df: pl.DataFrame, config: PlotConfig, col: str, **kwargs
    ):
        """Create histogram"""
        pass

    @abstractmethod
    def apply_annotations(self, fig, config: PlotConfig):
        """Apply annotations to figure"""
        pass

    @abstractmethod
    def export_plot(self, fig, filepath: str, config: PlotConfig):
        """Export plot to file"""
        pass

    @abstractmethod
    def apply_styling(self, fig, config: PlotConfig):
        """Apply styling to figure"""
        pass

    # Extended plots (with default fallback implementations)
    def create_box_plot(
        self,
        df: pl.DataFrame,
        config: PlotConfig,
        y_col: str,
        x_col: Optional[str] = None,
        **kwargs,
    ):
        """Create box plot - fallback implementation"""
        warnings.warn(
            f"{self.__class__.__name__} does not implement box_plot. Using fallback."
        )

        validate_dataframe(df, required_cols=[y_col].extend(x_col or []))

        if x_col and y_col:
            return self.create_scatter_plot(df, config, x_col, y_col, **kwargs)
        elif y_col:
            df_with_index = df.with_row_count("index")
            return self.create_scatter_plot(
                df_with_index, config, "index", y_col, **kwargs
            )
        else:
            raise ValueError("Need at least y_col for box plot fallback")

    def create_violin_plot(
        self,
        df: pl.DataFrame,
        config: PlotConfig,
        y_col: str,
        x_col: Optional[str] = None,
        **kwargs,
    ):
        """Create violin plot - fallback implementation"""
        warnings.warn(
            f"{self.__class__.__name__} does not implement violin_plot. Using box plot fallback."
        )
        validate_dataframe(df, required_cols=[y_col].extend(x_col or []))
        return self.create_box_plot(df, config, y_col, x_col, **kwargs)

    def create_heatmap(self, df: pl.DataFrame, config: PlotConfig, **kwargs):
        """Create heatmap - fallback implementation"""
        warnings.warn(
            f"{self.__class__.__name__} does not implement heatmap. Using fallback."
        )
        validate_dataframe(df)

        # Get numeric columns for fallback
        numeric_cols = [col for col in df.columns if df[col].dtype.is_numeric()]
        if len(numeric_cols) >= 2:
            return self.create_scatter_plot(
                df, config, numeric_cols[0], numeric_cols[1], **kwargs
            )
        else:
            raise ValueError("Need at least 2 numeric columns for heatmap fallback")

    def create_bubble_plot(self, df: pl.DataFrame, config: PlotConfig, **kwargs):
        """Create bubble plot - fallback implementation"""
        warnings.warn(
            f"{self.__class__.__name__} does not implement bubble_plot. Using scatter fallback."
        )
        validate_dataframe(df)
        numeric_cols = get_numeric_columns(df, min_count=2)
        return self.create_scatter_plot(
            df, config, numeric_cols[0], numeric_cols[1], **kwargs
        )

    def create_radar_chart(self, df: pl.DataFrame, config: PlotConfig, **kwargs):
        """Create radar chart - fallback implementation"""
        warnings.warn(
            f"{self.__class__.__name__} does not implement radar_chart. Using line fallback."
        )
        validate_dataframe(df)
        numeric_cols = get_numeric_columns(df, min_count=2)
        return self.create_line_plot(
            df, config, numeric_cols[0], numeric_cols[1], **kwargs
        )

    def create_parallel_coordinates(
        self, df: pl.DataFrame, config: PlotConfig, **kwargs
    ):
        """Create parallel coordinates - fallback implementation"""
        warnings.warn(
            f"{self.__class__.__name__} does not implement parallel_coordinates. Using line fallback."
        )
        validate_dataframe(df)
        numeric_cols = get_numeric_columns(df, min_count=2)
        return self.create_line_plot(
            df, config, numeric_cols[0], numeric_cols[1], **kwargs
        )

    def create_error_bar_plot(self, df: pl.DataFrame, config: PlotConfig, **kwargs):
        """Create error bar plot - fallback implementation"""
        warnings.warn(
            f"{self.__class__.__name__} does not implement error_bar_plot. Using scatter fallback."
        )
        validate_dataframe(df)
        numeric_cols = get_numeric_columns(df, min_count=2)
        return self.create_scatter_plot(
            df, config, numeric_cols[0], numeric_cols[1], **kwargs
        )

    def create_regression_plot(
        self, df: pl.DataFrame, config: PlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Create regression plot - fallback implementation"""
        warnings.warn(
            f"{self.__class__.__name__} does not implement regression_plot. Using scatter fallback."
        )
        validate_dataframe(df, required_cols=[x_col, y_col])
        return self.create_scatter_plot(df, config, x_col, y_col, **kwargs)

    def create_residual_plot(self, df: pl.DataFrame, config: PlotConfig, **kwargs):
        """Create residual plot - fallback implementation"""
        warnings.warn(
            f"{self.__class__.__name__} does not implement residual_plot. Using scatter fallback."
        )
        validate_dataframe(df)
        return self.create_scatter_plot(df, config, **kwargs)

    def create_qq_plot(self, df: pl.DataFrame, config: PlotConfig, **kwargs):
        """Create Q-Q plot - fallback implementation"""
        warnings.warn(
            f"{self.__class__.__name__} does not implement qq_plot. Using scatter fallback."
        )
        validate_dataframe(df)
        return self.create_scatter_plot(df, config, **kwargs)

    def create_density_plot(self, df: pl.DataFrame, config: PlotConfig, **kwargs):
        """Create density plot - fallback implementation"""
        warnings.warn(
            f"{self.__class__.__name__} does not implement density_plot. Using histogram fallback."
        )
        validate_dataframe(df)
        return self.create_histogram(df, config, **kwargs)

    def create_cdf_plot(self, df: pl.DataFrame, config: PlotConfig, **kwargs):
        """Create CDF plot - fallback implementation"""
        warnings.warn(
            f"{self.__class__.__name__} does not implement cdf_plot. Using line fallback."
        )
        validate_dataframe(df)
        return self.create_line_plot(df, config, **kwargs)

    def create_subplots_grid(self, df: pl.DataFrame, config: PlotConfig, **kwargs):
        """Create subplots grid - fallback implementation"""
        warnings.warn(
            f"{self.__class__.__name__} does not implement subplots_grid. Using single plot fallback."
        )
        validate_dataframe(df)
        return self.create_line_plot(df, config, **kwargs)

    def create_pair_plot(
        self, df: pl.DataFrame, config: PlotConfig, columns: List[str], **kwargs
    ):
        """Create pair plot - fallback implementation"""
        warnings.warn(
            f"{self.__class__.__name__} does not implement pair_plot. Using scatter fallback."
        )
        validate_dataframe(df, required_cols=columns)
        return self.create_scatter_plot(df, config, **kwargs)

    def create_facet_grid(self, df: pl.DataFrame, config: PlotConfig, **kwargs):
        """Create facet grid - fallback implementation"""
        warnings.warn(
            f"{self.__class__.__name__} does not implement facet_grid. Using single plot fallback."
        )
        validate_dataframe(df)
        return self.create_line_plot(df, config, **kwargs)

    def create_time_series_decomposition(
        self, df: pl.DataFrame, config: PlotConfig, **kwargs
    ):
        """Create time series decomposition - fallback implementation"""
        warnings.warn(
            f"{self.__class__.__name__} does not implement time_series_decomposition. Using line fallback."
        )
        validate_dataframe(df)
        return self.create_line_plot(df, config, **kwargs)

    def create_acf_pacf_plot(self, df: pl.DataFrame, config: PlotConfig, **kwargs):
        """Create ACF/PACF plot - fallback implementation"""
        warnings.warn(
            f"{self.__class__.__name__} does not implement acf_pacf_plot. Using line fallback."
        )
        validate_dataframe(df)
        return self.create_line_plot(df, config, **kwargs)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def validate_dataframe(df: pl.DataFrame, required_cols: Optional[List[str]] = None):
    """Validate DataFrame before plotting"""
    if len(df) == 0:
        raise ValueError("Cannot create plot with empty DataFrame")

    if required_cols:
        validate_dataframe_columns(df, required_cols)
    return True


def get_numeric_columns(df: pl.DataFrame, min_count: int = 1):
    """Get numeric columns with minimum count validation"""
    numeric_cols = [col for col in df.columns if df[col].dtype.is_numeric()]
    if len(numeric_cols) < min_count:
        raise ValueError(
            f"Need at least {min_count} numeric columns, found {len(numeric_cols)}"
        )
    return numeric_cols


def create_quick_config(
    title: str = "", xlabel: str = "", ylabel: str = "", **kwargs
) -> PlotConfig:
    """Create quick plot configuration"""
    return PlotConfig(title=title, xlabel=xlabel, ylabel=ylabel, **kwargs)


def validate_dataframe_columns(df: pl.DataFrame, required_cols: List[str]) -> bool:
    """Validate that DataFrame has required columns"""
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")
    return True


def get_optimal_figsize(
    data_aspect_ratio: float = 1.0, base_size: float = 8
) -> Tuple[float, float]:
    """Get optimal figure size based on data aspect ratio"""
    if data_aspect_ratio >= 1:
        return (base_size * data_aspect_ratio, base_size)
    else:
        return (base_size, base_size / data_aspect_ratio)
