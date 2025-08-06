"""
Complete Research Plot Library - Extended Chart Types
Comprehensive plotting utilities for scientific research
"""

from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import polars as pl
import numpy as np
import pandas as pd  # Imported for time series analysis functions
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from pandas.plotting import autocorrelation_plot  # Using pandas for ACF plot
from .utils import PlotConfig, MatplotlibBackend, PlotlyBackend


# Extended Matplotlib Backend with all plot types
class ExtendedMatplotlibBackend(MatplotlibBackend):
    """Extended Matplotlib backend with comprehensive plot types"""

    def create_bar_plot(
        self, df: pl.DataFrame, config: PlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Create bar plot"""
        self._setup_figure(config)
        colors = config.color_config.get_colors()

        self.ax.bar(
            df[x_col],
            df[y_col],
            color=colors[0],
            alpha=kwargs.get("alpha", 0.8),
            **kwargs,
        )

        self.apply_annotations(self.fig, config)
        return self.fig

    def create_histogram(
        self, df: pl.DataFrame, config: PlotConfig, col: str, **kwargs
    ):
        """Create histogram"""
        self._setup_figure(config)
        colors = config.color_config.get_colors()

        self.ax.hist(
            df[col],
            bins=kwargs.get("bins", 30),
            color=colors[0],
            alpha=kwargs.get("alpha", 0.7),
            edgecolor="black",
            **kwargs,
        )

        self.apply_annotations(self.fig, config)
        return self.fig

    def create_box_plot(
        self,
        df: pl.DataFrame,
        config: PlotConfig,
        y_col: str,
        x_col: Optional[str] = None,
        **kwargs,
    ):
        """Create box plot"""
        self._setup_figure(config)

        if x_col:
            # Grouped box plot
            data_groups = []
            labels = []
            for group in df[x_col].unique():
                group_data = df.filter(pl.col(x_col) == group)[y_col].to_list()
                data_groups.append(group_data)
                labels.append(str(group))

            bp = self.ax.boxplot(data_groups, labels=labels, patch_artist=True)
            colors = config.color_config.get_colors()
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
        else:
            # Single box plot
            bp = self.ax.boxplot(df[y_col].to_list(), patch_artist=True)
            colors = config.color_config.get_colors()
            bp["boxes"][0].set_facecolor(colors[0])

        self.apply_annotations(self.fig, config)
        return self.fig

    def create_violin_plot(
        self,
        df: pl.DataFrame,
        config: PlotConfig,
        y_col: str,
        x_col: Optional[str] = None,
        **kwargs,
    ):
        """Create violin plot"""
        self._setup_figure(config)

        if x_col:
            # Convert to format seaborn expects
            data_melted = df.select([x_col, y_col]).to_pandas()

            # --- FIX STARTS HERE ---
            # Get the exact number of categories to plot
            num_categories = df[x_col].n_unique()
            colors = config.color_config.get_colors()

            # Address the warnings by implementing the suggested changes
            sns.violinplot(
                data=data_melted,
                x=x_col,
                y=y_col,
                ax=self.ax,
                hue=x_col,  # FIX 1: Assign x to hue as recommended
                palette=colors[
                    :num_categories
                ],  # FIX 2: Slice palette to match category count
                legend=False,  # FIX 1: Disable the redundant legend
            )
        # --- FIX ENDS HERE ---

        else:
            # For a single violin, we can still customize the color for better aesthetics
            data_list = df[y_col].to_list()
            colors = config.color_config.get_colors()
            parts = self.ax.violinplot(data_list, showmedians=True)
            # Color the main body of the violin
            for pc in parts["bodies"]:
                pc.set_facecolor(colors[0])
                pc.set_edgecolor("black")
                pc.set_alpha(0.7)
            # Color the lines inside the violin plot for consistency
            for part_name in ("cbars", "cmins", "cmaxes", "cmedians"):
                if part_name in parts:
                    vp = parts[part_name]
                    vp.set_edgecolor("black")
                    vp.set_linewidth(1.5)

        self.apply_annotations(self.fig, config)
        return self.fig

    def create_heatmap(self, df: pl.DataFrame, config: PlotConfig, **kwargs):
        """Create heatmap"""
        self._setup_figure(config)

        # Convert to correlation matrix if needed
        if kwargs.get("correlation", False):
            numeric_cols = [
                col
                for col in df.columns
                if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
            ]
            corr_matrix = df.select(numeric_cols).to_pandas().corr()
            data_to_plot = corr_matrix
        else:
            data_to_plot = df.to_pandas()

        sns.heatmap(
            data_to_plot,
            ax=self.ax,
            cmap=kwargs.get("cmap", "viridis"),
            annot=kwargs.get("annot", True),
            fmt=kwargs.get("fmt", ".2f"),
        )

        self.apply_annotations(self.fig, config)
        return self.fig

    def create_bubble_plot(
        self,
        df: pl.DataFrame,
        config: PlotConfig,
        x_col: str,
        y_col: str,
        size_col: str,
        **kwargs,
    ):
        """Create bubble plot"""
        self._setup_figure(config)
        colors = config.color_config.get_colors()

        # Normalize sizes
        sizes = df[size_col].to_numpy()
        size_normalized = (sizes - sizes.min()) / (
            sizes.max() - sizes.min()
        ) * 1000 + 50

        scatter = self.ax.scatter(
            df[x_col],
            df[y_col],
            s=size_normalized,
            c=colors[0],
            alpha=kwargs.get("alpha", 0.6),
            edgecolors="black",
            linewidth=0.5,
        )

        # Add colorbar if color column provided
        if "color_col" in kwargs:
            color_data = df[kwargs["color_col"]].to_numpy()
            scatter = self.ax.scatter(
                df[x_col],
                df[y_col],
                s=size_normalized,
                c=color_data,
                alpha=kwargs.get("alpha", 0.6),
                cmap=kwargs.get("cmap", "viridis"),
            )
            plt.colorbar(scatter, ax=self.ax)

        self.apply_annotations(self.fig, config)
        return self.fig

    def create_radar_chart(
        self,
        df: pl.DataFrame,
        config: PlotConfig,
        category_col: str,
        value_cols: List[str],
        **kwargs,
    ):
        """Create radar chart"""
        # Radar chart requires polar projection
        fig = plt.figure(figsize=config.figsize, dpi=config.dpi)
        ax = fig.add_subplot(111, projection="polar")
        self.fig = fig
        self.ax = ax

        angles = np.linspace(0, 2 * np.pi, len(value_cols), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        colors = config.color_config.get_colors()

        for i, category in enumerate(df[category_col].unique()):
            category_data = df.filter(pl.col(category_col) == category)
            values = [category_data[col].item() for col in value_cols]
            values += values[:1]  # Complete the circle

            ax.plot(
                angles,
                values,
                "o-",
                linewidth=2,
                label=str(category),
                color=colors[i % len(colors)],
            )
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(value_cols)
        ax.set_title(config.title, fontsize=config.font_config["title_size"])
        if config.legend:
            ax.legend()

        return fig

    def create_parallel_coordinates(
        self,
        df: pl.DataFrame,
        config: PlotConfig,
        value_cols: List[str],
        class_col: Optional[str] = None,
        **kwargs,
    ):
        """Create parallel coordinates plot"""
        self._setup_figure(config)

        # Normalize data
        df_normalized = df.clone()
        for col in value_cols:
            col_data = df[col]
            min_val, max_val = col_data.min(), col_data.max()
            df_normalized = df_normalized.with_columns(
                ((pl.col(col) - min_val) / (max_val - min_val)).alias(col)
            )

        colors = config.color_config.get_colors()
        x_positions = list(range(len(value_cols)))

        if class_col:
            for i, class_val in enumerate(df[class_col].unique()):
                class_data = df_normalized.filter(pl.col(class_col) == class_val)
                for idx in range(len(class_data)):
                    row = class_data.row(idx)
                    y_values = [row[df.columns.index(col)] for col in value_cols]
                    self.ax.plot(
                        x_positions,
                        y_values,
                        color=colors[i % len(colors)],
                        alpha=kwargs.get("alpha", 0.7),
                        linewidth=kwargs.get("linewidth", 1),
                    )
        else:
            for idx in range(len(df_normalized)):
                row = df_normalized.row(idx)
                y_values = [row[df.columns.index(col)] for col in value_cols]
                self.ax.plot(
                    x_positions,
                    y_values,
                    color=colors[0],
                    alpha=kwargs.get("alpha", 0.7),
                    linewidth=kwargs.get("linewidth", 1),
                )

        self.ax.set_xticks(x_positions)
        self.ax.set_xticklabels(value_cols)
        self.apply_annotations(self.fig, config)
        return self.fig

    def create_error_bar_plot(
        self,
        df: pl.DataFrame,
        config: PlotConfig,
        x_col: str,
        y_col: str,
        error_col: str,
        **kwargs,
    ):
        """Create error bar plot"""
        self._setup_figure(config)
        colors = config.color_config.get_colors()

        self.ax.errorbar(
            df[x_col],
            df[y_col],
            yerr=df[error_col],
            fmt="o-",
            color=colors[0],
            capsize=kwargs.get("capsize", 5),
            capthick=kwargs.get("capthick", 2),
            **kwargs,
        )

        self.apply_annotations(self.fig, config)
        return self.fig

    def create_regression_plot(
        self, df: pl.DataFrame, config: PlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Create regression plot with confidence intervals"""
        self._setup_figure(config)
        colors = config.color_config.get_colors()

        x_data = df[x_col].drop_nulls().to_numpy()
        y_data = df[y_col].drop_nulls().to_numpy()

        # Scatter plot
        self.ax.scatter(x_data, y_data, color=colors[0], alpha=0.6)

        # Regression line using scipy
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
        line = slope * x_data + intercept
        self.ax.plot(
            x_data, line, color=colors[1], linewidth=2, label=f"R² = {r_value ** 2:.3f}"
        )

        # Confidence intervals
        if kwargs.get("ci", True):
            # Simple confidence interval calculation using scipy.stats
            residuals = y_data - line
            mse = np.mean(residuals**2)
            se = np.sqrt(
                mse
                * (
                    1 / len(x_data)
                    + (x_data - np.mean(x_data)) ** 2
                    / np.sum((x_data - np.mean(x_data)) ** 2)
                )
            )
            t_val = stats.t.ppf(0.975, len(x_data) - 2)  # 95% CI t-value
            ci = t_val * se

            sorted_indices = np.argsort(x_data)
            self.ax.fill_between(
                x_data[sorted_indices],
                (line - ci)[sorted_indices],
                (line + ci)[sorted_indices],
                alpha=0.3,
                color=colors[1],
            )

        if config.legend:
            self.ax.legend()

        self.apply_annotations(self.fig, config)
        return self.fig

    def create_residual_plot(
        self,
        df: pl.DataFrame,
        config: PlotConfig,
        x_col: str,
        y_col: str,
        predicted_col: str,
        **kwargs,
    ):
        """Create residual plot"""
        self._setup_figure(config)
        colors = config.color_config.get_colors()

        residuals = df[y_col] - df[predicted_col]

        self.ax.scatter(df[predicted_col], residuals, color=colors[0], alpha=0.6)
        self.ax.axhline(y=0, color="red", linestyle="--", linewidth=2)

        self.ax.set_xlabel("Predicted Values")
        self.ax.set_ylabel("Residuals")

        self.apply_annotations(self.fig, config)
        return self.fig

    def create_qq_plot(self, df: pl.DataFrame, config: PlotConfig, col: str, **kwargs):
        """Create Q-Q plot"""
        self._setup_figure(config)
        colors = config.color_config.get_colors()

        data = df[col].drop_nulls().to_numpy()
        stats.probplot(data, dist=kwargs.get("dist", "norm"), plot=self.ax)

        # Customize colors
        self.ax.get_lines()[0].set_markerfacecolor(colors[0])
        self.ax.get_lines()[0].set_markeredgecolor(colors[0])
        self.ax.get_lines()[1].set_color(colors[1])

        self.apply_annotations(self.fig, config)
        return self.fig

    def create_density_plot(
        self, df: pl.DataFrame, config: PlotConfig, col: str, **kwargs
    ):
        """Create density plot"""
        self._setup_figure(config)
        colors = config.color_config.get_colors()

        data = df[col].drop_nulls().to_numpy()

        # KDE plot using scipy
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 1000)
        density = kde(x_range)

        self.ax.plot(x_range, density, color=colors[0], linewidth=2)
        self.ax.fill_between(x_range, density, alpha=0.3, color=colors[0])

        self.apply_annotations(self.fig, config)
        return self.fig

    def create_subplots_grid(
        self, df: pl.DataFrame, config: PlotConfig, plot_configs: List[Dict], **kwargs
    ):
        """Create subplot grid"""
        nrows = kwargs.get("nrows", 2)
        ncols = kwargs.get("ncols", 2)

        fig, axes = plt.subplots(nrows, ncols, figsize=config.figsize, dpi=config.dpi)
        self.fig = fig

        if not isinstance(axes, np.ndarray):
            axes = [axes]
        elif axes.ndim > 1:
            axes = axes.flatten()

        for i, plot_config in enumerate(plot_configs[: len(axes)]):
            self.ax = axes[i]
            plot_type = plot_config["type"]
            plot_params = plot_config.get("params", {})

            if plot_type == "line":
                self.ax.plot(df[plot_params["x_col"]], df[plot_params["y_col"]])
            elif plot_type == "scatter":
                self.ax.scatter(df[plot_params["x_col"]], df[plot_params["y_col"]])
            elif plot_type == "hist":
                self.ax.hist(df[plot_params["col"]], bins=plot_params.get("bins", 30))

            self.ax.set_title(plot_config.get("title", ""))

        plt.tight_layout()
        return fig

    def create_pair_plot(
        self, df: pl.DataFrame, config: PlotConfig, columns: List[str], **kwargs
    ):
        """Create pair plot"""
        # Using seaborn for a more robust pair plot implementation
        pd_df = df.select(columns).to_pandas()
        pair_grid = sns.pairplot(pd_df, diag_kind="hist", plot_kws={"alpha": 0.6})
        self.fig = pair_grid.fig
        return self.fig

    def create_time_series_decomposition(
        self,
        df: pl.DataFrame,
        config: PlotConfig,
        time_col: str,
        value_col: str,
        **kwargs,
    ):
        """Create time series decomposition plot using pandas rolling averages."""
        # Convert to pandas for time series operations
        df_pandas = df.select([time_col, value_col]).to_pandas()
        df_pandas[time_col] = pd.to_datetime(df_pandas[time_col])
        df_pandas.set_index(time_col, inplace=True)
        series = df_pandas[value_col]

        period = kwargs.get("period")
        if period is None:
            raise ValueError(
                "A 'period' must be provided for time series decomposition."
            )

        # Decompose using pandas
        # 1. Trend: Centered moving average
        trend = series.rolling(window=period, center=True).mean()
        # 2. Seasonal: Average of detrended series for each seasonal step
        detrended = series - trend
        seasonal_indices = np.arange(len(series)) % period
        seasonal_means = detrended.groupby(seasonal_indices).mean()
        seasonal = np.tile(seasonal_means, len(series) // period + 1)[: len(series)]
        seasonal = pd.Series(seasonal, index=series.index)
        # 3. Residual
        resid = series - trend - seasonal

        # Create subplots
        fig, axes = plt.subplots(
            4, 1, figsize=(config.figsize[0], config.figsize[1] * 2), sharex=True
        )
        self.fig = fig
        colors = config.color_config.get_colors()

        axes[0].plot(series, color=colors[0])
        axes[0].set_title("Observed")
        axes[1].plot(trend, color=colors[1])
        axes[1].set_title("Trend")
        axes[2].plot(seasonal, color=colors[2])
        axes[2].set_title("Seasonal")
        axes[3].plot(resid, "o", markersize=2, color=colors[3 % len(colors)])
        axes[3].set_title("Residual")
        axes[3].axhline(y=0, color="black", linestyle="--", alpha=0.5)

        plt.tight_layout()
        return fig

    def create_acf_plot(self, df: pl.DataFrame, config: PlotConfig, col: str, **kwargs):
        """Create ACF plot using pandas.plotting.autocorrelation_plot."""
        self._setup_figure(config)
        data = df[col].to_pandas()

        # Use the robust pandas plotting utility for ACF
        autocorrelation_plot(
            data, ax=self.ax, color=config.color_config.get_colors()[0]
        )
        self.ax.set_title("Autocorrelation Function (ACF)")
        self.ax.set_xlabel("Lag")
        self.ax.set_ylabel("Autocorrelation")

        return self.fig


# Extended Plotly Backend
class ExtendedPlotlyBackend(PlotlyBackend):
    """Extended Plotly backend with comprehensive plot types"""

    def create_bar_plot(
        self, df: pl.DataFrame, config: PlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Create Plotly bar plot"""
        colors = config.color_config.get_colors()

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=df[x_col],
                y=df[y_col],
                marker_color=colors[0],
                name=kwargs.get("name", "Bar"),
            )
        )

        self._setup_plotly_layout(fig, config)
        self.apply_annotations(fig, config)
        return fig

    def create_histogram(
        self, df: pl.DataFrame, config: PlotConfig, col: str, **kwargs
    ):
        """Create Plotly histogram"""
        colors = config.color_config.get_colors()

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=df[col],
                nbinsx=kwargs.get("bins", 30),
                marker_color=colors[0],
                name=kwargs.get("name", "Histogram"),
            )
        )

        self._setup_plotly_layout(fig, config)
        self.apply_annotations(fig, config)
        return fig

    def create_box_plot(
        self,
        df: pl.DataFrame,
        config: PlotConfig,
        y_col: str,
        x_col: Optional[str] = None,
        **kwargs,
    ):
        """Create Plotly box plot"""
        fig = go.Figure()

        if x_col:
            for group in df[x_col].unique():
                group_data = df.filter(pl.col(x_col) == group)[y_col]
                fig.add_trace(
                    go.Box(y=group_data, name=str(group), boxpoints="outliers")
                )
        else:
            fig.add_trace(
                go.Box(
                    y=df[y_col],
                    name=kwargs.get("name", "Box Plot"),
                    boxpoints="outliers",
                )
            )

        self._setup_plotly_layout(fig, config)
        self.apply_annotations(fig, config)
        return fig

    def create_violin_plot(
        self,
        df: pl.DataFrame,
        config: PlotConfig,
        y_col: str,
        x_col: Optional[str] = None,
        **kwargs,
    ):
        """Create Plotly violin plot"""
        fig = go.Figure()

        if x_col:
            for group in df[x_col].unique():
                group_data = df.filter(pl.col(x_col) == group)[y_col]
                fig.add_trace(
                    go.Violin(y=group_data, name=str(group), box_visible=True)
                )
        else:
            fig.add_trace(
                go.Violin(
                    y=df[y_col],
                    name=kwargs.get("name", "Violin Plot"),
                    box_visible=True,
                )
            )

        self._setup_plotly_layout(fig, config)
        self.apply_annotations(fig, config)
        return fig

    def create_heatmap(self, df: pl.DataFrame, config: PlotConfig, **kwargs):
        """Create Plotly heatmap"""
        if kwargs.get("correlation", False):
            numeric_cols = [
                col
                for col in df.columns
                if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
            ]
            corr_matrix = df.select(numeric_cols).to_pandas().corr()

            fig = go.Figure(
                data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale=kwargs.get("colorscale", "Viridis"),
                    text=corr_matrix.values,
                    texttemplate="%{text:.2f}",
                    textfont={"size": 10},
                )
            )
        else:
            data_matrix = df.to_pandas()
            fig = go.Figure(
                data=go.Heatmap(
                    z=data_matrix.values,
                    x=data_matrix.columns,
                    y=data_matrix.index,
                    colorscale=kwargs.get("colorscale", "Viridis"),
                )
            )

        self._setup_plotly_layout(fig, config)
        self.apply_annotations(fig, config)
        return fig

    def create_bubble_plot(
        self,
        df: pl.DataFrame,
        config: PlotConfig,
        x_col: str,
        y_col: str,
        size_col: str,
        **kwargs,
    ):
        """Create Plotly bubble plot"""
        colors = config.color_config.get_colors()

        fig = go.Figure()

        color_data = df[kwargs["color_col"]] if "color_col" in kwargs else None

        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode="markers",
                marker=dict(
                    size=df[size_col],
                    color=color_data if color_data is not None else colors[0],
                    colorscale=kwargs.get("colorscale", "Viridis"),
                    showscale=color_data is not None,
                    line=dict(width=1, color="DarkSlateGrey"),
                    sizemode="diameter",
                    sizeref=2.0 * max(df[size_col].drop_nulls()) / (40.0**2),
                    sizemin=4,
                ),
                text=(
                    df[kwargs.get("text_col", x_col)] if "text_col" in kwargs else None
                ),
                name=kwargs.get("name", "Bubble Plot"),
            )
        )

        self._setup_plotly_layout(fig, config)
        self.apply_annotations(fig, config)
        return fig

    def create_radar_chart(
        self,
        df: pl.DataFrame,
        config: PlotConfig,
        category_col: str,
        value_cols: List[str],
        **kwargs,
    ):
        """Create Plotly radar chart"""
        fig = go.Figure()
        colors = config.color_config.get_colors()

        for i, category in enumerate(df[category_col].unique()):
            category_data = df.filter(pl.col(category_col) == category)
            values = [category_data[col].item() for col in value_cols]

            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=value_cols,
                    fill="toself",
                    name=str(category),
                    line_color=colors[i % len(colors)],
                )
            )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True, range=[0, max([df[col].max() for col in value_cols])]
                )
            ),
            showlegend=config.legend,
            title=config.title,
        )

        return fig


# Extended Research Plotter
class ExtendedResearchPlotter:
    """Extended research plotter with all chart types"""

    def __init__(self, backend: str = "matplotlib"):
        self.backend_name = backend
        self.backend = self._create_backend(backend)

    def _create_backend(self, backend_name: str):
        """Create extended backend instance"""
        if backend_name.lower() == "matplotlib":
            return ExtendedMatplotlibBackend()
        elif backend_name.lower() == "plotly":
            return ExtendedPlotlyBackend()
        else:
            raise ValueError(f"Unsupported backend: {backend_name}")

    def switch_backend(self, backend: str):
        """Switch plotting backend"""
        self.backend = self._create_backend(backend)
        self.backend_name = backend

    # Basic Statistical Charts
    def line_plot(
        self, df: pl.DataFrame, config: PlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Create line plot - time series data, trend analysis"""
        return self.backend.create_line_plot(df, config, x_col, y_col, **kwargs)

    def scatter_plot(
        self, df: pl.DataFrame, config: PlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Create scatter plot - correlation analysis, data distribution"""
        return self.backend.create_scatter_plot(df, config, x_col, y_col, **kwargs)

    def bar_plot(
        self, df: pl.DataFrame, config: PlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Create bar plot - categorical data comparison"""
        return self.backend.create_bar_plot(df, config, x_col, y_col, **kwargs)

    def histogram(self, df: pl.DataFrame, config: PlotConfig, col: str, **kwargs):
        """Create histogram - data distribution, frequency analysis"""
        return self.backend.create_histogram(df, config, col, **kwargs)

    def box_plot(
        self,
        df: pl.DataFrame,
        config: PlotConfig,
        y_col: str,
        x_col: Optional[str] = None,
        **kwargs,
    ):
        """Create box plot - data distribution, outlier detection"""
        return self.backend.create_box_plot(df, config, y_col, x_col, **kwargs)

    def violin_plot(
        self,
        df: pl.DataFrame,
        config: PlotConfig,
        y_col: str,
        x_col: Optional[str] = None,
        **kwargs,
    ):
        """Create violin plot - data distribution density"""
        return self.backend.create_violin_plot(df, config, y_col, x_col, **kwargs)

    # Multi-dimensional Data Visualization
    def heatmap(self, df: pl.DataFrame, config: PlotConfig, **kwargs):
        """Create heatmap - correlation matrix, data matrix visualization"""
        return self.backend.create_heatmap(df, config, **kwargs)

    def bubble_plot(
        self,
        df: pl.DataFrame,
        config: PlotConfig,
        x_col: str,
        y_col: str,
        size_col: str,
        **kwargs,
    ):
        """Create bubble plot - three-dimensional data relationships"""
        return self.backend.create_bubble_plot(
            df, config, x_col, y_col, size_col, **kwargs
        )

    def radar_chart(
        self,
        df: pl.DataFrame,
        config: PlotConfig,
        category_col: str,
        value_cols: List[str],
        **kwargs,
    ):
        """Create radar chart - multi-indicator comparison"""
        return self.backend.create_radar_chart(
            df, config, category_col, value_cols, **kwargs
        )

    def parallel_coordinates(
        self,
        df: pl.DataFrame,
        config: PlotConfig,
        value_cols: List[str],
        class_col: Optional[str] = None,
        **kwargs,
    ):
        """Create parallel coordinates plot - high-dimensional data visualization"""
        return self.backend.create_parallel_coordinates(
            df, config, value_cols, class_col, **kwargs
        )

    # Scientific Specialized Charts
    def error_bar_plot(
        self,
        df: pl.DataFrame,
        config: PlotConfig,
        x_col: str,
        y_col: str,
        error_col: str,
        **kwargs,
    ):
        """Create error bar plot - experimental data uncertainty"""
        return self.backend.create_error_bar_plot(
            df, config, x_col, y_col, error_col, **kwargs
        )

    def regression_plot(
        self, df: pl.DataFrame, config: PlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Create regression plot - fitting curves and confidence intervals"""
        return self.backend.create_regression_plot(df, config, x_col, y_col, **kwargs)

    def residual_plot(
        self,
        df: pl.DataFrame,
        config: PlotConfig,
        x_col: str,
        y_col: str,
        predicted_col: str,
        **kwargs,
    ):
        """Create residual plot - model diagnostics"""
        return self.backend.create_residual_plot(
            df, config, x_col, y_col, predicted_col, **kwargs
        )

    def qq_plot(self, df: pl.DataFrame, config: PlotConfig, col: str, **kwargs):
        """Create Q-Q plot - distribution testing"""
        return self.backend.create_qq_plot(df, config, col, **kwargs)

    def density_plot(self, df: pl.DataFrame, config: PlotConfig, col: str, **kwargs):
        """Create density plot - probability density distribution"""
        return self.backend.create_density_plot(df, config, col, **kwargs)

    # Multi-subplot Combinations
    def subplots_grid(
        self, df: pl.DataFrame, config: PlotConfig, plot_configs: List[Dict], **kwargs
    ):
        """Create subplot grid - multiple related chart combinations"""
        return self.backend.create_subplots_grid(df, config, plot_configs, **kwargs)

    def pair_plot(
        self, df: pl.DataFrame, config: PlotConfig, columns: List[str], **kwargs
    ):
        """Create pair plot - pairwise variable relationships"""
        return self.backend.create_pair_plot(df, config, columns, **kwargs)

    def facet_grid(
        self,
        df: pl.DataFrame,
        config: PlotConfig,
        plot_type: str,
        x_col: str,
        y_col: str,
        facet_col: str,
        **kwargs,
    ):
        """Create facet grid - grouped display by category"""
        # Implementation for faceted plots
        facet_values = df[facet_col].unique().to_list()
        n_facets = len(facet_values)
        ncols = kwargs.get("ncols", min(3, n_facets))
        nrows = (n_facets + ncols - 1) // ncols

        if self.backend_name == "matplotlib":
            fig, axes = plt.subplots(
                nrows, ncols, figsize=config.figsize, dpi=config.dpi
            )
            if not isinstance(axes, np.ndarray):
                axes = [axes]
            elif axes.ndim > 1:
                axes = axes.flatten()

            colors = config.color_config.get_colors()

            for i, facet_val in enumerate(facet_values):
                if i < len(axes):
                    ax = axes[i]
                    facet_data = df.filter(pl.col(facet_col) == facet_val)

                    if plot_type == "scatter":
                        ax.scatter(
                            facet_data[x_col],
                            facet_data[y_col],
                            color=colors[0],
                            alpha=0.7,
                        )
                    elif plot_type == "line":
                        ax.plot(facet_data[x_col], facet_data[y_col], color=colors[0])
                    elif plot_type == "bar":
                        ax.bar(facet_data[x_col], facet_data[y_col], color=colors[0])

                    ax.set_title(f"{facet_col} = {facet_val}")
                    ax.set_xlabel(config.xlabel)
                    ax.set_ylabel(config.ylabel)

            # Hide unused subplots
            for i in range(len(facet_values), len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            return fig
        else:
            # Plotly implementation
            fig = make_subplots(
                rows=nrows,
                cols=ncols,
                subplot_titles=[f"{facet_col} = {val}" for val in facet_values],
            )

            colors = config.color_config.get_colors()

            for i, facet_val in enumerate(facet_values):
                row = i // ncols + 1
                col = i % ncols + 1
                facet_data = df.filter(pl.col(facet_col) == facet_val)

                if plot_type == "scatter":
                    trace = go.Scatter(
                        x=facet_data[x_col],
                        y=facet_data[y_col],
                        mode="markers",
                        marker_color=colors[0],
                        name=f"{facet_val}",
                    )
                elif plot_type == "line":
                    trace = go.Scatter(
                        x=facet_data[x_col],
                        y=facet_data[y_col],
                        mode="lines",
                        line_color=colors[0],
                        name=f"{facet_val}",
                    )
                elif plot_type == "bar":
                    trace = go.Bar(
                        x=facet_data[x_col],
                        y=facet_data[y_col],
                        marker_color=colors[0],
                        name=f"{facet_val}",
                    )

                fig.add_trace(trace, row=row, col=col)

            fig.update_layout(title_text=config.title, showlegend=False)
            return fig

    # Time Series Specialized
    def time_series_decomposition(
        self,
        df: pl.DataFrame,
        config: PlotConfig,
        time_col: str,
        value_col: str,
        **kwargs,
    ):
        """Create time series decomposition plot - trend, seasonality, residual"""
        # This check assumes Plotly backend might be missing this advanced method.
        if not hasattr(self.backend, "create_time_series_decomposition"):
            raise NotImplementedError(
                f"Time series decomposition not implemented for {self.backend_name} backend."
            )
        return self.backend.create_time_series_decomposition(
            df, config, time_col, value_col, **kwargs
        )

    def acf_plot(self, df: pl.DataFrame, config: PlotConfig, col: str, **kwargs):
        """Create ACF plot - time series analysis"""
        # This check assumes Plotly backend might be missing this advanced method.
        if not hasattr(self.backend, "create_acf_plot"):
            raise NotImplementedError(
                f"ACF plot not implemented for {self.backend_name} backend."
            )
        return self.backend.create_acf_plot(df, config, col, **kwargs)

    # Utility methods
    def export(self, fig, filepath: str, config: PlotConfig):
        """Export plot to file"""
        self.backend.export_plot(fig, filepath, config)


# Data generators for testing
class AdvancedDataGenerator:
    """Advanced data generators for different scientific scenarios"""

    @staticmethod
    def generate_experimental_data(n_groups=3, n_per_group=20):
        """Generate experimental data with groups and errors"""
        data = []
        for i in range(n_groups):
            group_mean = 10 + i * 5
            values = np.random.normal(group_mean, 2, n_per_group)
            errors = np.random.uniform(0.5, 2, n_per_group)

            for j, (val, err) in enumerate(zip(values, errors)):
                data.append(
                    {
                        "group": f"Group_{i + 1}",
                        "subject": j + 1,
                        "value": val,
                        "error": err,
                        "replicate": np.random.choice([1, 2, 3]),
                    }
                )

        return pl.DataFrame(data)

    @staticmethod
    def generate_dose_response_data():
        """Generate dose-response curve data"""
        doses = np.logspace(-3, 2, 20)  # 0.001 to 100

        # Hill equation: Response = Bottom + (Top-Bottom)/(1 + (IC50/dose)^Hill)
        bottom, top, ic50, hill = 5, 95, 10, 1.5
        response = bottom + (top - bottom) / (1 + (ic50 / doses) ** hill)
        response += np.random.normal(0, 3, len(doses))  # Add noise

        return pl.DataFrame(
            {
                "dose": doses,
                "response": response,
                "log_dose": np.log10(doses),
                "replicate": np.tile([1, 2, 3], len(doses) // 3 + 1)[: len(doses)],
            }
        )

    @staticmethod
    def generate_multivariate_data(n_features=5, n_samples=100):
        """Generate multivariate data with correlations"""
        # Create correlation matrix
        rho = 0.3
        corr_matrix = np.full((n_features, n_features), rho)
        np.fill_diagonal(corr_matrix, 1.0)

        # Generate correlated data
        data = np.random.multivariate_normal(
            mean=np.zeros(n_features), cov=corr_matrix, size=n_samples
        )

        feature_names = [f"feature_{i + 1}" for i in range(n_features)]
        df_dict = {name: data[:, i] for i, name in enumerate(feature_names)}
        df_dict["cluster"] = np.random.choice(["A", "B", "C"], n_samples)

        return pl.DataFrame(df_dict)

    @staticmethod
    def generate_time_series_with_trend():
        """Generate complex time series with trend, seasonality, and noise"""
        n_points = 365 * 2  # 2 years of daily data
        dates = pd.date_range("2022-01-01", periods=n_points, freq="D")

        # Trend component
        trend = np.linspace(100, 150, n_points)

        # Seasonal components
        yearly_seasonal = 10 * np.sin(2 * np.pi * np.arange(n_points) / 365.25)
        weekly_seasonal = 3 * np.sin(2 * np.pi * np.arange(n_points) / 7)

        # Noise
        noise = np.random.normal(0, 2, n_points)

        # Combine components
        values = trend + yearly_seasonal + weekly_seasonal + noise

        return pl.DataFrame(
            {
                "date": dates,
                "value": values,
                "trend": trend,
                "yearly_seasonal": yearly_seasonal,
                "weekly_seasonal": weekly_seasonal,
                "noise": noise,
            }
        )
