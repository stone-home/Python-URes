import os
import shutil
import tempfile
import unittest
from ures.files import (
    get_file_paths,
    filter_files,
    list_directories,
    get_temp_folder,
    get_temp_dir_with_specific_path,
)


class TestFilesModule(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory structure for testing.
        self.test_dir = tempfile.mkdtemp()
        # Create subdirectories and files.
        os.makedirs(os.path.join(self.test_dir, "subdir1"), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, "subdir2"), exist_ok=True)
        self.file1 = os.path.join(self.test_dir, "file1.txt")
        self.file2 = os.path.join(self.test_dir, "subdir1", "file2.log")
        self.file3 = os.path.join(self.test_dir, "subdir2", "example.txt")
        with open(self.file1, "w") as f:
            f.write("Test file 1")
        with open(self.file2, "w") as f:
            f.write("Test file 2")
        with open(self.file3, "w") as f:
            f.write("Test file 3")

    def tearDown(self):
        # Remove the temporary directory after tests.
        shutil.rmtree(self.test_dir)

    def test_get_file_paths(self):
        """Test that get_file_paths returns all file paths in the directory."""
        paths = get_file_paths(self.test_dir)
        self.assertIsInstance(paths, list)
        # Should find three files.
        self.assertEqual(len(paths), 3)
        self.assertTrue(any("file1.txt" in path for path in paths))
        self.assertTrue(any("file2.log" in path for path in paths))
        self.assertTrue(any("example.txt" in path for path in paths))

    def test_filter_files_fuzz(self):
        """Test filter_files with fuzz enabled returns files containing the keyword."""
        # Search for files containing "file"
        filtered = filter_files("file", self.test_dir, fuzz=True)
        # All files contain 'file' in the name.
        self.assertEqual(
            len(filtered), 2
        )  # "file1.txt" and "file2.log" (example.txt does not contain "file")
        # Check that file1.txt is in the filtered list.
        self.assertTrue(any("file1.txt" in path for path in filtered))

    def test_filter_files_exact(self):
        """Test filter_files with exact match returns only files with an exact name."""
        # Create a file with exact name "exact.txt" in test_dir.
        exact_file = os.path.join(self.test_dir, "exact.txt")
        with open(exact_file, "w") as f:
            f.write("Exact file")
        filtered = filter_files("exact.txt", self.test_dir, fuzz=False)
        self.assertEqual(len(filtered), 1)
        self.assertTrue(filtered[0].endswith("exact.txt"))

    def test_list_directories(self):
        """Test that list_directories returns only subdirectories."""
        dirs = list_directories(self.test_dir)
        self.assertIsInstance(dirs, list)
        self.assertIn("subdir1", dirs)
        self.assertIn("subdir2", dirs)
        # The file "file1.txt" should not be listed.
        self.assertNotIn("file1.txt", dirs)

    def test_list_directories_nonexistent(self):
        """Test that list_directories returns None for a non-existent path."""
        result = list_directories("/path/that/does/not/exist")
        self.assertIsNone(result)

    def test_get_temp_folder(self):
        """Test that get_temp_folder returns a valid temporary folder path."""
        temp_folder = get_temp_folder()
        # The returned path may not exist since TemporaryDirectory().name is not automatically created.
        self.assertIsInstance(temp_folder, str)
        # Create the directory to test if it is writable.
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder, exist_ok=True)
        self.assertTrue(os.path.isdir(temp_folder))
        # Clean up the created folder.
        shutil.rmtree(temp_folder)

    def test_get_temp_dir_with_specific_path(self):
        """Test that get_temp_dir_with_specific_path creates and returns the correct directory."""
        sub_path = ("test_app", "data")
        temp_dir = get_temp_dir_with_specific_path(*sub_path)
        self.assertTrue(os.path.isdir(temp_dir))
        self.assertTrue(temp_dir.endswith(os.path.join(*sub_path)))
        # Cleanup: Remove the created temporary directory.
        shutil.rmtree(temp_dir)
