import pytest
import pandas as pd
import math  # Needed for checking NaN values

# Assume your function is in a file named 'your_module.py'
# If it's in the same file, you don't need the import.
# from your_module import align_and_pad_lists


# --- Code Under Test ---
# Included here for completeness, normally it would be imported.
def align_and_pad_lists(args: list) -> list:
    """
    Aligns lists in a list to the length of the longest list
    by forward-filling shorter lists.
    """
    if not args:
        # Let max() handle the empty list case to raise ValueError
        pass

    all_lengths = [len(lst) for lst in args]
    if not all_lengths and args:  # Input like [[]] or [[], []]
        max_length = 0
    elif not args:  # Input is []
        max_length = max([])  # This line will raise ValueError
    else:
        max_length = max(all_lengths)

    if max_length == 0:
        return [[] for _ in args]

    aligned_list = [
        pd.Series(lst).reindex(range(max_length), method="ffill").tolist()
        for lst in args
    ]
    return aligned_list


# --- Helper Function for NaN Comparison ---
# Can remain outside the classes or be included as a static method if preferred
def assert_list_equal_with_nan(list1, list2):
    """Asserts two lists are equal, handling NaN values correctly."""
    assert len(list1) == len(
        list2
    ), f"Lists have different lengths: {len(list1)} != {len(list2)}"
    for i in range(len(list1)):
        item1, item2 = list1[i], list2[i]
        is_nan1 = isinstance(item1, float) and math.isnan(item1)
        is_nan2 = isinstance(item2, float) and math.isnan(item2)
        if is_nan1 and is_nan2:
            continue
        assert item1 == item2, f"Items at index {i} differ: {item1} != {item2}"


# --- Test Classes for Grouping ---


# Enhanced Helper Function
def assert_list_equal_flexible(list1, list2):
    """
    Asserts two lists are equal, handling NaN/None and int/float equivalence.
    """
    assert len(list1) == len(
        list2
    ), f"Lists have different lengths: {len(list1)} != {len(list2)}"
    for i in range(len(list1)):
        item1, item2 = list1[i], list2[i]

        # Check if both are NaN
        is_nan1 = isinstance(item1, float) and math.isnan(item1)
        is_nan2 = isinstance(item2, float) and math.isnan(item2)
        if is_nan1 and is_nan2:
            continue  # Both are NaN, consider equal

        # Check if one is NaN and the other is None
        is_none1 = item1 is None
        is_none2 = item2 is None
        if (is_nan1 and is_none2) or (is_none1 and is_nan2):
            continue  # Allow None to equal NaN for this comparison

        # Check for numeric equivalence (e.g., 1 == 1.0)
        # Avoid comparing None directly with numbers using this method
        if not (is_none1 or is_none2):
            try:
                # Use tolerance if comparing floats directly? For now, direct float compare.
                if float(item1) == float(item2):
                    continue  # Treat 1 and 1.0 as equal
            except (ValueError, TypeError):
                # Not comparable as floats (e.g., strings involved), proceed to direct comparison
                pass

        # Fallback to direct comparison for other types or if above checks didn't match
        assert (
            item1 == item2
        ), f"Items at index {i} differ: ({type(item1).__name__}){item1} != ({type(item2).__name__}){item2}"


class TestPaddingBehavior:
    """Tests focused on the core padding logic and type handling."""

    @pytest.mark.parametrize(
        "test_id, input_data, expected_output",
        [
            # --- Keep expected outputs conceptually correct (using None, int) ---
            (
                "basic_varying_lengths",
                [[1, 2], [3, 4, 5], [6]],
                [[1, 2, 2], [3, 4, 5], [6, 6, 6]],
            ),
            ("fill_from_last_element", [[1, 2, 3], [4]], [[1, 2, 3], [4, 4, 4]]),
            ("fill_at_start_of_reindex", [[5], [1, 2, 3]], [[5, 5, 5], [1, 2, 3]]),
            (
                "internal_none_values",
                [[1, None, 2], [3, 4]],
                [[1, None, 2], [3, 4, 4]],
            ),  # Expect None
            (
                "mixed_data_types",
                [[1, 2], ["a", "b", "c"], [None, 4.5]],
                [[1, 2, 2], ["a", "b", "c"], [None, 4.5, 4.5]],
            ),  # Expect None
        ],
        ids=lambda x: x if isinstance(x, str) else "",
    )
    def test_padding_and_types(self, test_id, input_data, expected_output):
        """Parametrized test for various padding scenarios and types."""
        actual_output = align_and_pad_lists(input_data)

        # --- Use the flexible helper function in a loop ---
        assert len(actual_output) == len(
            expected_output
        ), "Outer lists have different lengths."
        for i in range(len(expected_output)):
            assert_list_equal_flexible(
                actual_output[i], expected_output[i]
            )  # Use the enhanced helper


class TestEdgeCases:
    """Tests focused on edge cases like no padding needed or zero length lists."""

    @pytest.mark.parametrize(
        "test_id, input_data, expected_output",
        [
            ("all_same_length", [[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]),
            ("single_list", [[10, 20, 30]], [[10, 20, 30]]),
        ],
        ids=lambda x: x if isinstance(x, str) else "",
    )
    def test_no_padding_needed(self, test_id, input_data, expected_output):
        """Tests scenarios where no padding adjustment should be needed."""
        actual_output = align_and_pad_lists(input_data)
        assert actual_output == expected_output

    @pytest.mark.parametrize(
        "test_id, input_data, expected_output",
        [
            ("all_empty_inner", [[], [], []], [[], [], []]),
            ("single_empty_inner", [[]], [[]]),
        ],
        ids=lambda x: x if isinstance(x, str) else "",
    )
    def test_zero_max_length(self, test_id, input_data, expected_output):
        """Tests scenarios where the maximum length is 0 (only empty lists)."""
        actual_output = align_and_pad_lists(input_data)
        assert actual_output == expected_output


class TestSpecialHandling:
    """Tests focused on specific outcomes like NaN generation or error handling."""

    def test_empty_inner_list_produces_nan(self):
        """Tests alignment with an empty inner list results in NaNs."""
        input_data = [[1, 2], [], [3, 4, 5]]
        expected_output = [
            [1, 2, 2],
            [float("nan"), float("nan"), float("nan")],
            [3, 4, 5],
        ]
        actual_output = align_and_pad_lists(input_data)
        # Use the helper function defined outside the class
        assert len(actual_output) == len(expected_output)
        assert_list_equal_with_nan(actual_output[0], expected_output[0])
        assert_list_equal_with_nan(actual_output[1], expected_output[1])
        assert_list_equal_with_nan(actual_output[2], expected_output[2])

    def test_empty_input_list_raises_valueerror(self):
        """Tests that providing an empty list [] raises ValueError."""
        input_data = []
        with pytest.raises(ValueError):
            align_and_pad_lists(input_data)
