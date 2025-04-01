from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Union, Optional
from .data_structure.memory import AbsMemoryBlock
from .string import format_memory


class MemoryBlock(AbsMemoryBlock, ABC):
    @property
    @abstractmethod
    def alloc_time(self) -> Union[int, float]:
        pass

    @alloc_time.setter
    @abstractmethod
    def alloc_time(self, value: Union[int, float]):
        pass

    @property
    @abstractmethod
    def free_time(self) -> Optional[Union[int, float]]:
        pass

    @free_time.setter
    @abstractmethod
    def free_time(self, value: Optional[Union[int, float]]):
        pass

    @property
    @abstractmethod
    def comment(self) -> str:
        pass



class MemoryOperator:
    def __init__(self, blocks: list[MemoryBlock]):
        self.blocks = blocks

    def visualize_memory_lifecycle(self):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        memory_blocks = self.blocks

        # Calculate the earliest allocation time
        min_alloc_time = min(block.alloc_time for block in memory_blocks)

        # Subtract the earliest allocation time from the alloc and free times of each memory block
        shifted_memory_blocks = []
        for block in memory_blocks:
            shifted_block = deepcopy(block)  # Copy the dictionary
            shifted_block.alloc_time -= min_alloc_time
            if block.free_time is not None:
                shifted_block.free_time -= min_alloc_time
            shifted_memory_blocks.append(shifted_block)

        # Calculate the maximum display time: use the maximum release time among all memory blocks with release times, and extend the display range if there is permanent allocation
        free_times = [block.free_time for block in shifted_memory_blocks if block.free_time is not None]
        max_free_time = max(free_times) if free_times else 0
        max_alloc_time = max(block.alloc_time for block in shifted_memory_blocks)
        max_time = max(max_free_time, max_alloc_time) + 2  # Add some margin

        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(10, len(memory_blocks) / 5))
        rect_height = 0.8

        for i, block in enumerate(shifted_memory_blocks):
            alloc = block.alloc_time
            # If the memory block has no release time, draw it until max_time
            free = block.free_time if block.free_time is not None else max_time
            size = block.bytes
            # Use different colors to distinguish between permanent and non-permanent allocation: skyblue for those with release times, lightgreen for those without (permanent)
            color = 'skyblue' if block.free_time is not None else 'lightgreen'

            # Create a rectangle, with the bottom-left corner at (alloc, i), width (free - alloc), and fixed height
            rect = patches.Rectangle((alloc, i), free - alloc, rect_height,
                                     facecolor=color, edgecolor='black', alpha=0.8)
            ax.add_patch(rect)

            # Annotate the memory size in the middle of the rectangle
            ax.text((alloc + free) / 2, i + rect_height / 2, f"Size: {format_memory(size)}",
                    ha='center', va='center', fontsize=10, color='black')

        # Set the x-axis range
        ax.set_xlim(0, max_time)
        # Dynamically set the y-axis range based on the number of memory blocks, adding a margin of 0.5 at the top and bottom for a compact and beautiful graph
        ax.set_ylim(-0.5, len(shifted_memory_blocks) - 0.5)
        ax.set_xlabel("Time (shifted)")
        ax.set_ylabel("Memory Block")
        ax.set_title("Memory Block Allocation Timeline (Shifted)")

        # Set the y-axis ticks to the index of each memory block
        ax.set_yticks([i for i in range(len(shifted_memory_blocks))])
        ax.set_yticklabels([f"{i.comment}" for i in shifted_memory_blocks])

        # Draw x-axis grid lines
        ax.grid(axis='x', linestyle='--', alpha=0.7)

        plt.show()