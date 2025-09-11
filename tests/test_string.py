import pytest
import unittest
from ures.string import (
    zettelkasten_id,
    unique_id,
    format_memory,
    capitalize_string,
    string2date,
)


class TestStringFunctions(unittest.TestCase):
    def test_zettelkasten_id_format(self):
        """Test that zettelkasten_id returns a string in the expected format."""
        zk_id = zettelkasten_id()
        pattern = r"^[a-f0-9]{9}\.[a-f0-9]{11}$"
        self.assertIsInstance(zk_id, str)
        self.assertRegex(zk_id, pattern)

    def test_zettelkasten_id_uniqueness(self):
        """Test that two consecutive zettelkasten_id calls produce different IDs."""
        id1 = zettelkasten_id()
        id2 = zettelkasten_id()
        self.assertNotEqual(id1, id2)

    def test_unique_id_length_and_format(self):
        """Test that unique_id returns a 32-character hexadecimal string."""
        uid = unique_id()
        self.assertIsInstance(uid, str)
        self.assertEqual(len(uid), 32)
        # Check that all characters are valid hexadecimal digits.
        self.assertTrue(all(c in "0123456789abcdef" for c in uid))

    def test_unique_id_uniqueness(self):
        """Test that two consecutive unique_id calls produce different IDs."""
        uid1 = unique_id()
        uid2 = unique_id()
        self.assertNotEqual(uid1, uid2)

    def test_format_memory_bytes(self):
        """Test format_memory with a value less than 1KB."""
        self.assertEqual(format_memory(500), "500 B")

    def test_format_memory_kb(self):
        """Test format_memory converts bytes to kilobytes correctly."""
        self.assertEqual(format_memory(1024), "1.00 KB")
        self.assertEqual(format_memory(1536), "1.50 KB")  # 1.5 KB

    def test_format_memory_mb(self):
        """Test format_memory converts bytes to megabytes correctly."""
        self.assertEqual(format_memory(1048576), "1.00 MB")  # 1 MB

    def test_format_memory_gb(self):
        """Test format_memory converts bytes to gigabytes correctly."""
        self.assertEqual(format_memory(1073741824), "1.00 GB")  # 1 GB

    def test_format_memory_none(self):
        """Test that passing None returns '0 bytes'."""
        self.assertEqual(format_memory(None), "0 bytes")

    def test_capitalize_string_default_separator(self):
        """Test capitalize_string with the default space separator."""
        input_str = "hello world"
        expected = "Hello World"
        self.assertEqual(capitalize_string(input_str), expected)

    def test_capitalize_string_custom_separator(self):
        """Test capitalize_string with a custom separator."""
        input_str = "john-doe"
        expected = "John-Doe"
        self.assertEqual(capitalize_string(input_str, separator="-"), expected)

    def test_capitalize_string_empty(self):
        """Test capitalize_string with an empty string returns an empty string."""
        self.assertEqual(capitalize_string(""), "")


class TestString2Date:
    """Test class for string2date function"""

    def test_full_date_format_yyyy_mm_dd(self):
        """Test parsing complete date format YYYY-MM-DD"""
        result = string2date("2023-12-25")
        expected = {"year": 2023, "month": "Dec", "day": 25}
        assert result == expected

        result = string2date("2024-01-01")
        expected = {"year": 2024, "month": "Jan", "day": 1}
        assert result == expected

        result = string2date("2022-07-15")
        expected = {"year": 2022, "month": "Jul", "day": 15}
        assert result == expected

    def test_year_month_format_yyyy_mm(self):
        """Test parsing year-month format YYYY-MM"""
        result = string2date("2023-12")
        expected = {"year": 2023, "month": "Dec", "day": None}
        assert result == expected

        result = string2date("2024-01")
        expected = {"year": 2024, "month": "Jan", "day": None}
        assert result == expected

        result = string2date("2022-06")
        expected = {"year": 2022, "month": "Jun", "day": None}
        assert result == expected

    def test_year_only_format_yyyy(self):
        """Test parsing year-only format YYYY"""
        result = string2date("2023")
        expected = {"year": 2023, "month": None, "day": None}
        assert result == expected

        result = string2date("1999")
        expected = {"year": 1999, "month": None, "day": None}
        assert result == expected

        result = string2date("2050")
        expected = {"year": 2050, "month": None, "day": None}
        assert result == expected

    def test_edge_case_years(self):
        """Test edge cases for year validation"""
        # Minimum valid year
        result = string2date("0001")
        expected = {"year": 1, "month": None, "day": None}
        assert result == expected

        # Maximum valid year
        result = string2date("9999")
        expected = {"year": 9999, "month": None, "day": None}
        assert result == expected

    def test_leap_year_dates(self):
        """Test dates in leap years"""
        # February 29 in a leap year
        result = string2date("2024-02-29")
        expected = {"year": 2024, "month": "Feb", "day": 29}
        assert result == expected

        # February in a leap year (month only)
        result = string2date("2024-02")
        expected = {"year": 2024, "month": "Feb", "day": None}
        assert result == expected

    def test_all_months(self):
        """Test all 12 months to ensure correct abbreviations"""
        month_tests = [
            ("2023-01", "Jan"),
            ("2023-02", "Feb"),
            ("2023-03", "Mar"),
            ("2023-04", "Apr"),
            ("2023-05", "May"),
            ("2023-06", "Jun"),
            ("2023-07", "Jul"),
            ("2023-08", "Aug"),
            ("2023-09", "Sep"),
            ("2023-10", "Oct"),
            ("2023-11", "Nov"),
            ("2023-12", "Dec"),
        ]

        for date_str, expected_month in month_tests:
            result = string2date(date_str)
            assert result["month"] == expected_month
            assert result["year"] == 2023
            assert result["day"] is None

    def test_invalid_date_formats(self):
        """Test various invalid date formats"""
        invalid_inputs = [
            "2023/12/25",  # Wrong separator
            "25-12-2023",  # Wrong order
            "12-25-2023",  # MM-DD-YYYY format
            "2023-13-01",  # Invalid month
            "2023-02-30",  # Invalid day for February
            "2023-04-31",  # Invalid day for April
            "23-12-25",  # 2-digit year
            "abc-def-ghi",  # Non-numeric
        ]

        expected = {"year": None, "month": None, "day": None}

        for invalid_input in invalid_inputs:
            result = string2date(invalid_input)
            assert result == expected, f"Failed for input: {invalid_input}"

    def test_invalid_year_formats(self):
        """Test invalid year formats"""
        invalid_years = [
            "23",  # 2-digit year
            "123",  # 3-digit year
            "12345",  # 5-digit year
            "0000",  # Year zero
            "10000",  # Year beyond range
            "-2023",  # Negative year
            "abcd",  # Non-numeric
            "",  # Empty string
            "20a3",  # Mixed characters
        ]

        expected = {"year": None, "month": None, "day": None}

        for invalid_year in invalid_years:
            result = string2date(invalid_year)
            assert result == expected, f"Failed for input: {invalid_year}"

    def test_empty_and_none_inputs(self):
        """Test edge cases with empty or None inputs"""
        # Empty string
        result = string2date("")
        expected = {"year": None, "month": None, "day": None}
        assert result == expected

        # Whitespace
        result = string2date("   ")
        expected = {"year": None, "month": None, "day": None}
        assert result == expected

    def test_boundary_dates(self):
        """Test boundary dates like end/start of months and years"""
        # Last day of year
        result = string2date("2023-12-31")
        expected = {"year": 2023, "month": "Dec", "day": 31}
        assert result == expected

        # First day of year
        result = string2date("2023-01-01")
        expected = {"year": 2023, "month": "Jan", "day": 1}
        assert result == expected

        # Last day of February (non-leap year)
        result = string2date("2023-02-28")
        expected = {"year": 2023, "month": "Feb", "day": 28}
        assert result == expected

    def test_leading_zeros(self):
        """Test dates with leading zeros"""
        result = string2date("2023-01-01")
        expected = {"year": 2023, "month": "Jan", "day": 1}
        assert result == expected

        result = string2date("2023-09-05")
        expected = {"year": 2023, "month": "Sep", "day": 5}
        assert result == expected

    @pytest.mark.parametrize(
        "date_string,expected",
        [
            ("2023-12-25", {"year": 2023, "month": "Dec", "day": 25}),
            ("2023-12", {"year": 2023, "month": "Dec", "day": None}),
            ("2023", {"year": 2023, "month": None, "day": None}),
            ("invalid", {"year": None, "month": None, "day": None}),
            ("2023-13-01", {"year": None, "month": None, "day": None}),
            ("0001", {"year": 1, "month": None, "day": None}),
            ("9999", {"year": 9999, "month": None, "day": None}),
        ],
    )
    def test_parametrized_inputs(self, date_string, expected):
        """Parametrized test for various input combinations"""
        result = string2date(date_string)
        assert result == expected
