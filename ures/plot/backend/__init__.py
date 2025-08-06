import polars as pl
from typing import List, Dict, Any, Optional
from .interactive import InteractiveBackend
from .research import ResearchBackend
from .core import (
    Annotation,
    AnnotationCollection,
    PlotBackend,
    PlotConfig,
    ColorConfig,
    ColorScheme,
    validate_dataframe,
    create_quick_config,
    get_optimal_figsize,
)
from .deps import (
    HAS_SEABORN,
    HAS_PLOTLY,
    HAS_SCIPY,
    HAS_SKLEARN,
    HAS_STATSMODELS,
    HAS_HOLOVIEWS,
    check_dependencies,
    check_optional_dependency,
)


# ============================================================================
# MAIN RESEARCH PLOTTER CLASS
# ============================================================================


class ResearchPlotter:
    """Main research plotting class with dual backend support"""

    def __init__(self, backend: str = "research"):
        """
        Initialize plotter with specified backend

        Args:
                backend: "research" (matplotlib+seaborn) or "interactive" (plotly)
        """
        self.backend_name = backend
        self.backend = self._create_backend(backend)

    def _create_backend(self, backend_name: str) -> PlotBackend:
        """Create backend instance with validation"""
        if backend_name.lower() in ["research", "matplotlib"]:
            return ResearchBackend()
        elif backend_name.lower() in ["interactive", "plotly"]:
            return InteractiveBackend()
        else:
            raise ValueError(
                f"Unsupported backend: {backend_name}. Use 'research' or 'interactive'"
            )

    def switch_backend(self, backend: str):
        """Switch plotting backend"""
        self.backend = self._create_backend(backend)
        self.backend_name = backend

    # ====== CORE PLOTTING METHODS ======

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

    def bar_plot(
        self, df: pl.DataFrame, config: PlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Create bar plot"""
        return self.backend.create_bar_plot(df, config, x_col, y_col, **kwargs)

    def histogram(self, df: pl.DataFrame, config: PlotConfig, col: str, **kwargs):
        """Create histogram"""
        return self.backend.create_histogram(df, config, col, **kwargs)

    # ====== EXTENDED PLOTTING METHODS ======

    def box_plot(
        self,
        df: pl.DataFrame,
        config: PlotConfig,
        y_col: str,
        x_col: Optional[str] = None,
        **kwargs,
    ):
        """Create box plot"""
        return self.backend.create_box_plot(df, config, y_col, x_col, **kwargs)

    def violin_plot(
        self,
        df: pl.DataFrame,
        config: PlotConfig,
        y_col: str,
        x_col: Optional[str] = None,
        **kwargs,
    ):
        """Create violin plot"""
        return self.backend.create_violin_plot(df, config, y_col, x_col, **kwargs)

    def heatmap(self, df: pl.DataFrame, config: PlotConfig, **kwargs):
        """Create heatmap"""
        return self.backend.create_heatmap(df, config, **kwargs)

    def regression_plot(
        self, df: pl.DataFrame, config: PlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Create regression plot"""
        return self.backend.create_regression_plot(df, config, x_col, y_col, **kwargs)

    def pair_plot(
        self, df: pl.DataFrame, config: PlotConfig, columns: List[str], **kwargs
    ):
        """Create pair plot"""
        return self.backend.create_pair_plot(df, config, columns, **kwargs)

    # ====== UTILITY METHODS ======

    def export(self, fig, filepath: str, config: PlotConfig):
        """Export figure"""
        self.backend.export_plot(fig, filepath, config)

    def get_backend_info(self) -> Dict[str, Any]:
        """Get current backend information"""
        return {
            "name": self.backend_name,
            "type": type(self.backend).__name__,
            "has_seaborn": HAS_SEABORN,
            "has_plotly": HAS_PLOTLY,
            "has_scipy": HAS_SCIPY,
        }
