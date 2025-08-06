from typing import Dict, List, Tuple, Optional, Union, Any
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import warnings
from .core import PlotBackend, PlotConfig, validate_dataframe
from .deps import HAS_SEABORN, HAS_SCIPY, HAS_STATSMODELS


# Import optional dependencies
if HAS_SEABORN:
    import seaborn as sns

if HAS_SCIPY:
    from scipy.interpolate import interp1d

if HAS_STATSMODELS:
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        from statsmodels.stats.diagnostic import acf, pacf
    except ImportError:
        HAS_STATSMODELS = False


# ============================================================================
# MATPLOTLIB + SEABORN RESEARCH BACKEND (主力)
# ============================================================================


class ResearchBackend(PlotBackend):
    """Matplotlib + Seaborn Research Plotting Backend (主力研究图表生成)"""

    def __init__(self):
        self.fig = None
        self.ax = None

    def _setup_figure(self, config: PlotConfig):
        """Enhanced figure setup with Seaborn integration"""
        # Use Seaborn if available for better defaults
        if HAS_SEABORN:
            if config.style == "whitegrid":
                sns.set_style("whitegrid")
            elif config.style in ["darkgrid", "white", "dark", "ticks"]:
                sns.set_style(config.style)
            else:
                sns.set_style("whitegrid")
        else:
            # Fallback to matplotlib styles
            available_styles = plt.style.available
            if config.style in available_styles:
                plt.style.use(config.style)
            else:
                plt.style.use("default")
                plt.rcParams["axes.grid"] = True

        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        self._set_labels(config)
        self._apply_enhanced_styling(config)

    def _set_labels(self, config: PlotConfig):
        """Set axis labels with units"""
        xlabel = config.xlabel
        ylabel = config.ylabel

        if config.x_unit and config.unit_position in ["label", "both"]:
            xlabel = f"{xlabel} ({config.x_unit})" if xlabel else f"({config.x_unit})"
        if config.y_unit and config.unit_position in ["label", "both"]:
            ylabel = f"{ylabel} ({config.y_unit})" if ylabel else f"({config.y_unit})"

        self.ax.set_xlabel(xlabel, fontsize=config.font_config["label_size"])
        self.ax.set_ylabel(ylabel, fontsize=config.font_config["label_size"])
        self.ax.set_title(config.title, fontsize=config.font_config["title_size"])

    def _apply_enhanced_styling(self, config: PlotConfig):
        """Apply enhanced styling"""
        if config.grid:
            self.ax.grid(True, alpha=0.3)

        if not config.show_spines:
            for spine in self.ax.spines.values():
                spine.set_visible(False)
        else:
            for spine in self.ax.spines.values():
                spine.set_alpha(config.spine_alpha)

        self.ax.tick_params(labelsize=config.font_config["tick_size"])

    def _apply_smoothing(self, x_data, y_data, config: PlotConfig):
        """Data smoothing with SciPy"""
        if not config.smooth or not HAS_SCIPY:
            return x_data, y_data

        try:
            if config.smooth_method == "spline" and len(x_data) > 3:
                f = interp1d(x_data, y_data, kind="cubic")
                x_smooth = np.linspace(x_data.min(), x_data.max(), len(x_data) * 3)
                y_smooth = f(x_smooth)
                return x_smooth, y_smooth
            else:
                # Moving average fallback
                window = max(3, int(len(x_data) * config.smooth_factor))
                y_smooth = np.convolve(y_data, np.ones(window) / window, mode="same")
                return x_data, y_smooth
        except Exception as e:
            warnings.warn(f"Smoothing failed: {e}")
            return x_data, y_data

    # ====== CORE IMPLEMENTATIONS ======

    def create_line_plot(
        self, df: pl.DataFrame, config: PlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Enhanced line plot with Seaborn styling"""
        validate_dataframe(df, required_cols=[x_col, y_col])
        self._setup_figure(config)

        colors = config.color_config.get_colors()
        x_data = df[x_col].to_numpy()
        y_data = df[y_col].to_numpy()

        x_smooth, y_smooth = self._apply_smoothing(x_data, y_data, config)

        line_kwargs = {
            "color": kwargs.get("color", colors[0]),
            "linewidth": kwargs.get("linewidth", 2),
            "alpha": kwargs.get("alpha", 1.0),
            "linestyle": kwargs.get("linestyle", "-"),
            "marker": kwargs.get("marker", None),
            "markersize": kwargs.get("markersize", 4),
            "label": kwargs.get("label", None),
        }

        line_kwargs = {k: v for k, v in line_kwargs.items() if v is not None}
        self.ax.plot(x_smooth, y_smooth, **line_kwargs)

        if config.legend and kwargs.get("label"):
            self.ax.legend(fontsize=config.font_config["legend_size"])

        self.apply_annotations(self.fig, config)
        self.apply_styling(self.fig, config)
        return self.fig

    def create_scatter_plot(
        self, df: pl.DataFrame, config: PlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Enhanced scatter plot with Seaborn styling"""
        validate_dataframe(df, required_cols=[x_col, y_col])
        self._setup_figure(config)

        colors = config.color_config.get_colors()

        if HAS_SEABORN and kwargs.get("use_seaborn", True):
            # Use seaborn for enhanced scatter plot
            self.ax = sns.scatterplot(
                data=df.to_pandas(),
                x=x_col,
                y=y_col,
                color=kwargs.get("color", colors[0]),
                s=kwargs.get("s", 50),
                alpha=kwargs.get("alpha", 0.7),
                ax=self.ax,
            )
        else:
            # Fallback to matplotlib
            scatter_kwargs = {
                "c": kwargs.get("c", colors[0]),
                "s": kwargs.get("s", 50),
                "alpha": kwargs.get("alpha", 0.7),
                "edgecolors": kwargs.get("edgecolors", "none"),
                "marker": kwargs.get("marker", "o"),
            }
            self.ax.scatter(df[x_col], df[y_col], **scatter_kwargs)

        self.apply_annotations(self.fig, config)
        self.apply_styling(self.fig, config)
        return self.fig

    def create_bar_plot(
        self, df: pl.DataFrame, config: PlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Enhanced bar plot with Seaborn styling"""
        validate_dataframe(df, required_cols=[x_col, y_col])
        self._setup_figure(config)

        colors = config.color_config.get_colors()

        if HAS_SEABORN and kwargs.get("use_seaborn", True):
            sns.barplot(
                data=df.to_pandas(),
                x=x_col,
                y=y_col,
                color=kwargs.get("color", colors[0]),
                alpha=kwargs.get("alpha", 0.8),
                ax=self.ax,
            )
        else:
            bar_kwargs = {
                "color": kwargs.get("color", colors[0]),
                "alpha": kwargs.get("alpha", 0.8),
                "edgecolor": kwargs.get("edgecolor", "black"),
                "linewidth": kwargs.get("linewidth", 0.5),
            }
            self.ax.bar(df[x_col], df[y_col], **bar_kwargs)

        self.apply_annotations(self.fig, config)
        self.apply_styling(self.fig, config)
        return self.fig

    def create_histogram(
        self, df: pl.DataFrame, config: PlotConfig, col: str, **kwargs
    ):
        """Enhanced histogram with Seaborn styling"""
        validate_dataframe(df, required_cols=[col])
        self._setup_figure(config)

        colors = config.color_config.get_colors()

        if HAS_SEABORN and kwargs.get("use_seaborn", True):
            sns.histplot(
                data=df.to_pandas(),
                x=col,
                bins=kwargs.get("bins", 30),
                color=kwargs.get("color", colors[0]),
                alpha=kwargs.get("alpha", 0.7),
                kde=kwargs.get("kde", False),
                ax=self.ax,
            )
        else:
            hist_kwargs = {
                "bins": kwargs.get("bins", 30),
                "color": kwargs.get("color", colors[0]),
                "alpha": kwargs.get("alpha", 0.7),
                "edgecolor": kwargs.get("edgecolor", "black"),
                "density": kwargs.get("density", False),
            }
            self.ax.hist(df[col], **hist_kwargs)

        self.apply_annotations(self.fig, config)
        self.apply_styling(self.fig, config)
        return self.fig

    # ====== EXTENDED IMPLEMENTATIONS ======

    def create_box_plot(
        self,
        df: pl.DataFrame,
        config: PlotConfig,
        y_col: str,
        x_col: Optional[str] = None,
        **kwargs,
    ):
        """Enhanced box plot with Seaborn"""
        validate_dataframe(df, required_cols=[x_col, y_col])
        self._setup_figure(config)

        if HAS_SEABORN:
            sns.boxplot(
                data=df.to_pandas(),
                x=x_col,
                y=y_col,
                hue=x_col,  # Assign x_col to hue
                palette=config.color_config.get_colors(),
                legend=False,  # Add this to remove the legend
                ax=self.ax,
            )
        else:
            # Simple matplotlib boxplot
            if x_col:
                groups = df.group_by(x_col)[y_col].apply(list)
                self.ax.boxplot([group for group in groups], labels=groups.index)
            else:
                self.ax.boxplot(df[y_col].to_list())

        self.apply_annotations(self.fig, config)
        self.apply_styling(self.fig, config)
        return self.fig

    def create_violin_plot(
        self,
        df: pl.DataFrame,
        config: PlotConfig,
        y_col: str,
        x_col: Optional[str] = None,
        **kwargs,
    ):
        """Enhanced violin plot with Seaborn"""
        validate_dataframe(df, required_cols=[y_col].extend([x_col] if x_col else []))
        self._setup_figure(config)

        if HAS_SEABORN:
            # Convert to format seaborn expects
            data_melted = df.select([x_col, y_col]).to_pandas()

            # Address the warnings by implementing the suggested changes
            sns.violinplot(
                data=data_melted,
                x=x_col,
                y=y_col,
                ax=self.ax,
                hue=x_col,  # FIX 1: Assign x to hue as recommended
                palette=config.color_config.get_colors(),
                legend=False,  # FIX 1: Disable the redundant legend
            )
        else:
            try:
                warnings.warn(
                    "Seaborn not available. Using MatPlot Violin plot instead of Seaborn Violin."
                )
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
            except Exception as e:
                warnings.warn(
                    "Seaborn not available. Using box plot instead of violin plot."
                )
                return self.create_box_plot(df, config, y_col, x_col, **kwargs)

        self.apply_annotations(self.fig, config)
        self.apply_styling(self.fig, config)
        return self.fig

    def create_heatmap(self, df: pl.DataFrame, config: PlotConfig, **kwargs):
        """Enhanced heatmap with Seaborn"""
        validate_dataframe(df)
        self._setup_figure(config)

        # Convert to correlation matrix if needed
        if kwargs.get("correlation", True):
            corr_data = df.select(
                [pl.col(c) for c in df.columns if df[c].dtype.is_numeric()]
            ).corr()
        else:
            corr_data = df

        if HAS_SEABORN:
            sns.heatmap(
                corr_data.to_pandas(),
                annot=kwargs.get("annot", True),
                cmap=kwargs.get("cmap", "viridis"),
                center=kwargs.get("center", 0),
                ax=self.ax,
            )
        else:
            # Simple matplotlib heatmap
            im = self.ax.imshow(corr_data.to_numpy(), cmap="viridis", aspect="auto")
            self.fig.colorbar(im, ax=self.ax)

        self.apply_annotations(self.fig, config)
        self.apply_styling(self.fig, config)
        return self.fig

    def create_regression_plot(
        self, df: pl.DataFrame, config: PlotConfig, x_col: str, y_col: str, **kwargs
    ):
        """Enhanced regression plot with Seaborn"""
        validate_dataframe(df, required_cols=[x_col, y_col])
        self._setup_figure(config)

        if HAS_SEABORN:
            sns.regplot(
                data=df.to_pandas(),
                x=x_col,
                y=y_col,
                scatter_kws={"alpha": kwargs.get("alpha", 0.6)},
                line_kws={"color": kwargs.get("line_color", "red")},
                ax=self.ax,
            )
        else:
            # Simple scatter + line
            self.ax.scatter(df[x_col], df[y_col], alpha=kwargs.get("alpha", 0.6))
            # Add simple trend line
            z = np.polyfit(df[x_col], df[y_col], 1)
            p = np.poly1d(z)
            self.ax.plot(df[x_col], p(df[x_col]), "r--", alpha=0.8)

        self.apply_annotations(self.fig, config)
        self.apply_styling(self.fig, config)
        return self.fig

    def create_pair_plot(
        self, df: pl.DataFrame, config: PlotConfig, columns: List[str], **kwargs
    ):
        """Enhanced pair plot with Seaborn"""
        validate_dataframe(df, required_cols=columns)
        if HAS_SEABORN:
            # Use seaborn's pairplot
            g = sns.pairplot(
                df.select(columns).to_pandas(),
                diag_kind=kwargs.get("diag_kind", "hist"),
                plot_kws={"alpha": kwargs.get("alpha", 0.6)},
            )
            return g.fig
        else:
            # Simple fallback - just create scatter plot of first two columns
            warnings.warn(
                "Seaborn not available. Creating simple scatter plot instead."
            )
            if len(columns) >= 2:
                return self.create_scatter_plot(
                    df, config, columns[0], columns[1], **kwargs
                )
            else:
                return self.create_histogram(df, config, columns[0], **kwargs)

    def apply_annotations(self, fig, config: PlotConfig):
        """Enhanced annotation application"""
        for ann in config.annotations.annotations:
            try:
                if ann.annotation_type == "text":
                    self.ax.annotate(
                        ann.text,
                        (ann.x, ann.y),
                        fontsize=ann.fontsize,
                        color=ann.color,
                        xycoords=ann.xycoords,
                        ha=ann.ha,
                        va=ann.va,
                        rotation=ann.rotation,
                        alpha=ann.alpha,
                    )
                elif ann.annotation_type == "arrow":
                    self.ax.annotate(
                        ann.text,
                        xy=getattr(ann, "xytext", (ann.x, ann.y)),
                        xytext=(ann.x, ann.y),
                        arrowprops=ann.arrow_props
                        or {"arrowstyle": "->", "color": "red"},
                        fontsize=ann.fontsize,
                        color=ann.color,
                        ha=ann.ha,
                        va=ann.va,
                        rotation=ann.rotation,
                        alpha=ann.alpha,
                    )
                elif ann.annotation_type == "box":
                    self.ax.annotate(
                        ann.text,
                        (ann.x, ann.y),
                        bbox=ann.bbox_props,
                        fontsize=ann.fontsize,
                        color=ann.color,
                        ha=ann.ha,
                        va=ann.va,
                        rotation=ann.rotation,
                        alpha=ann.alpha,
                    )
            except Exception as e:
                warnings.warn(f"Failed to add annotation: {e}")

    def apply_styling(self, fig, config: PlotConfig):
        """Apply final styling touches"""
        if config.tight_layout:
            try:
                fig.tight_layout()
            except Exception as e:
                warnings.warn(f"tight_layout failed: {e}")

    def export_plot(self, fig, filepath: str, config: PlotConfig):
        """Enhanced plot export"""
        try:
            fig.savefig(
                filepath,
                format=config.export_format,
                dpi=config.export_dpi,
                transparent=config.export_transparent,
                bbox_inches="tight",
                facecolor="white" if not config.export_transparent else "none",
            )
            plt.close(fig)
        except Exception as e:
            warnings.warn(f"Failed to export plot: {e}")
