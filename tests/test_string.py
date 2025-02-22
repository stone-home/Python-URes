import re
import unittest
from ures.string import zettelkasten_id, unique_id, format_memory, capitalize_string


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
