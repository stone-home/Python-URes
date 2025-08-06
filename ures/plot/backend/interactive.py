import polars as pl
import warnings
from .core import PlotBackend, PlotConfig, validate_dataframe
from .deps import HAS_PLOTLY


if HAS_PLOTLY:
    import plotly.graph_objects as go

# ============================================================================
# PLOTLY INTERACTIVE BACKEND
# ============================================================================


class InteractiveBackend(PlotBackend):
    """Plotly Interactive Plotting Backend (交互式图表辅助)"""

    def __init__(self):
        if not HAS_PLOTLY:
            raise ImportError("Plotly not available. Install with: pip install plotly")

    def create_line_plot(
        self, df: pl.DataFrame, config: PlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Interactive line plot with Plotly"""
        validate_dataframe(df, required_cols=[x_col, y_col])
        colors = config.color_config.get_colors()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode="lines",
                line=dict(color=colors[0], width=kwargs.get("linewidth", 2)),
                name=kwargs.get("name", "Line"),
                opacity=kwargs.get("alpha", 1.0),
            )
        )

        self._setup_plotly_layout(fig, config)
        self.apply_annotations(fig, config)
        self.apply_styling(fig, config)
        return fig

    def create_scatter_plot(
        self, df: pl.DataFrame, config: PlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Interactive scatter plot with Plotly"""
        validate_dataframe(df, required_cols=[x_col, y_col])
        colors = config.color_config.get_colors()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode="markers",
                marker=dict(
                    color=colors[0],
                    size=kwargs.get("size", 8),
                    opacity=kwargs.get("alpha", 0.7),
                ),
                name=kwargs.get("name", "Scatter"),
            )
        )

        self._setup_plotly_layout(fig, config)
        self.apply_annotations(fig, config)
        self.apply_styling(fig, config)
        return fig

    def create_bar_plot(
        self, df: pl.DataFrame, config: PlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Interactive bar plot with Plotly"""
        validate_dataframe(df, required_cols=[x_col, y_col])
        colors = config.color_config.get_colors()

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=df[x_col],
                y=df[y_col],
                marker=dict(color=colors[0], opacity=kwargs.get("alpha", 0.8)),
                name=kwargs.get("name", "Bar"),
            )
        )

        self._setup_plotly_layout(fig, config)
        self.apply_annotations(fig, config)
        self.apply_styling(fig, config)
        return fig

    def create_histogram(
        self, df: pl.DataFrame, config: PlotConfig, col: str, **kwargs
    ):
        """Interactive histogram with Plotly"""
        validate_dataframe(df, required_cols=[col])
        colors = config.color_config.get_colors()

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=df[col],
                nbinsx=kwargs.get("bins", 30),
                marker=dict(color=colors[0], opacity=kwargs.get("alpha", 0.7)),
                name=kwargs.get("name", "Histogram"),
            )
        )

        self._setup_plotly_layout(fig, config)
        self.apply_annotations(fig, config)
        self.apply_styling(fig, config)
        return fig

    def _setup_plotly_layout(self, fig, config: PlotConfig):
        """Enhanced Plotly layout setup"""
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
            showlegend=config.legend,
        )

        if config.grid:
            fig.update_xaxes(showgrid=True)
            fig.update_yaxes(showgrid=True)

    def apply_annotations(self, fig, config: PlotConfig):
        """Interactive annotations for Plotly"""
        for ann in config.annotations.annotations:
            try:
                fig.add_annotation(
                    text=ann.text,
                    x=ann.x,
                    y=ann.y,
                    showarrow=ann.showarrow,
                    arrowhead=ann.arrowhead,
                    font=dict(size=ann.fontsize, color=ann.color),
                    opacity=ann.alpha,
                    textangle=ann.rotation,
                )
            except Exception as e:
                warnings.warn(f"Failed to add Plotly annotation: {e}")

    def apply_styling(self, fig, config: PlotConfig):
        """Apply Plotly styling"""
        # Additional Plotly-specific styling
        pass

    def export_plot(self, fig, filepath: str, config: PlotConfig):
        """Enhanced Plotly export"""
        try:
            if config.export_format.lower() == "html":
                fig.write_html(filepath)
            else:
                # Requires kaleido package
                fig.write_image(
                    filepath,
                    format=config.export_format,
                    width=config.figsize[0] * config.export_dpi,
                    height=config.figsize[1] * config.export_dpi,
                )
        except Exception as e:
            warnings.warn(
                f"Failed to export Plotly plot: {e}. Try: pip install kaleido"
            )
