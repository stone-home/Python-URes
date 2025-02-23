import datetime
import unittest
from ures.timedate import datetime_converter, timestamp_converter, time_now


class TestTimeDateFunctions(unittest.TestCase):
    def test_datetime_converter_iso8601(self):
        """Test converting a datetime to ISO 8601 format."""
        dt = datetime.datetime(2020, 1, 1, 12, 0, 0)
        result = datetime_converter(dt, iso8601=True)
        expected = "2020-01-01T12:00:00Z"
        self.assertEqual(result, expected)

    def test_datetime_converter_custom_format(self):
        """Test converting a datetime using a custom format."""
        dt = datetime.datetime(2020, 1, 1, 12, 0, 0)
        result = datetime_converter(dt, iso8601=False, format="%Y/%m/%d %H:%M")
        expected = "2020/01/01 12:00"
        self.assertEqual(result, expected)

    def test_timestamp_converter_iso8601(self):
        """Test converting a Unix timestamp to ISO 8601 format."""
        # 2020-01-01 12:00:00 corresponds to 1577880000 seconds since epoch.
        result = timestamp_converter(1577880000, iso8601=True)
        expected = "2020-01-01T12:00:00Z"
        self.assertEqual(result, expected)

    def test_timestamp_converter_custom_format(self):
        """Test converting a Unix timestamp using a custom format."""
        result = timestamp_converter(1577880000, iso8601=False, format="%d/%m/%Y %H:%M")
        expected = "01/01/2020 12:00"
        self.assertEqual(result, expected)

    def test_time_now_iso8601(self):
        """Test that time_now returns a string ending with 'Z' when iso8601 is True."""
        current_time = time_now(iso8601=True)
        self.assertIsInstance(current_time, str)
        self.assertTrue(current_time.endswith("Z"))

    def test_time_now_custom_format(self):
        """Test that time_now returns a correctly formatted string using a custom format."""
        current_time = time_now(iso8601=False, format="%H:%M:%S")
        # Ensure that the result matches the pattern of HH:MM:SS (24-hour format)
        pattern = r"^\d{2}:\d{2}:\d{2}$"
        self.assertRegex(current_time, pattern)
