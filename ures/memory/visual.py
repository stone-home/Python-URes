import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go
from typing import List, Union
from ..data_process import align_and_pad_lists
from ..data_structure.memory import Memory
from ..string import format_memory


class MemoryVisualization:
    def memory_change_plot_l(
        self,
        title: str,
        data: list[list],
        labels: list[str],
        layout_conf: dict = None,
    ):
        l_data = align_and_pad_lists(*data)

        min_length = min(len(data), len(labels))
        # Create a new figure
        fig = go.Figure()

        # Add traces for tensor and segment memory usage
        for i in range(min_length):
            x_data = l_data[i]
            xlabel = labels[i]
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(x_data))),
                    y=x_data,
                    mode="lines",
                    name=xlabel,
                )
            )

        # Update the layout with axis labels and title
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Memory Usage",
            **layout_conf or {},
        )

        # Display the figure
        fig.show()

    def display_memory_lifecycle_pydantic(
        self, memory_blocks: List[Memory]
    ):  # Updated type hint
        """
        Visualizes memory block allocation timelines using Matplotlib,
        accepting a list of Pydantic Memory objects.

        Args:
            memory_blocks: A list of Memory objects. Blocks without a valid numeric
                           'alloc_time' will be ignored.
        """

        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(10, len(memory_blocks) / 5))
        rect_height = 0.8

        # Calculate the earliest allocation time
        min_alloc_time = min(
            block.alloc_time for block in memory_blocks if block.alloc_time is not None
        )
        max_free_time = max(
            block.free_time for block in memory_blocks if block.free_time is not None
        )

        # Subtract the earliest allocation time from the alloc and free times of each memory block
        for index, block in enumerate(memory_blocks):
            colour = "skyblue"
            shifted_block = block.model_copy(deep=True)
            if shifted_block.alloc_time is None:
                shifted_block.alloc_time = min_alloc_time
                shifted_block.free_time = max_free_time
                colour = "lightgreen"
            else:
                shifted_block.alloc_time -= min_alloc_time
                if block.free_time is None:
                    shifted_block.free_time = max_free_time
                    colour = "lightgreen"
                else:
                    shifted_block.free_time -= min_alloc_time

            rect = patches.Rectangle(
                (shifted_block.alloc_time, index),
                shifted_block.duration,
                rect_height,
                facecolor=colour,
                edgecolor="black",
                alpha=0.8,
            )
            ax.add_patch(rect)
            ax.text(
                (shifted_block.alloc_time + shifted_block.free_time) / 2,
                index + rect_height / 2,
                f"Size: {format_memory(shifted_block.bytes)}",
                ha="left",
                va="center",
                fontsize=10,
                color="black",
            )

        # Set the x-axis range
        ax.set_xlim(0, max_free_time)
        # Dynamically set the y-axis range based on the number of memory blocks, adding a margin of 0.5 at the top and bottom for a compact and beautiful graph
        ax.set_ylim(-0.5, len(memory_blocks) - 0.5)
        ax.set_xlabel("Time (shifted)")
        ax.set_ylabel("Memory Block")
        ax.set_title("Memory Block Allocation Timeline (Shifted)")

        # Set the y-axis ticks to the index of each memory block
        ax.set_yticks([i for i in range(len(memory_blocks))])
        ax.set_yticklabels([f"{i['name']}" for i in memory_blocks])

        # Draw x-axis grid lines
        ax.grid(axis="x", linestyle="--", alpha=0.7)

        plt.show()

    def plot_memory_stats(self):
        """Plot memory states for all segments

        Examples:
            The input of states should be formed as below:
            states = [
                {
                    "device": Device,
                    "stream": Stream,
                    "total: int,
                    "free": int,
                    "active": int,
                    'blocks': [Block, Block, Block, ...]
                }
            ]

        Args:
            states (List[dict]): A list of memory states
            filename (str): The filename to save the plot

        Returns:
            None

        """
        pass
