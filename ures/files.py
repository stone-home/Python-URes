import os
import tempfile
from typing import List, Optional


def get_file_paths(directory: str) -> List[str]:
    """
    Retrieve all file paths within the specified directory and its subdirectories.

    This function walks through the given directory recursively and returns a list of
    absolute file paths for every file found.

    Args:
        directory (str): The directory path to search in.

    Returns:
        List[str]: A list containing the absolute paths of all files found.

    Example:
        >>> paths = get_file_paths("/path/to/directory")
        >>> isinstance(paths, list)
        True
    """
    file_paths = []
    # Walk through the directory and its subdirectories.
    for root, _, files in os.walk(directory):
        for file in files:
            # Join the root path with the file name to get the absolute path.
            file_path = os.path.join(root, file)
            file_paths.append(str(file_path))
    return file_paths


def filter_files(part_file_name: str, directory: str, fuzz: bool = True) -> List[str]:
    """
    Retrieve file paths in the specified directory that match a given file name pattern.

    This function filters file paths within a directory by checking if the file name contains
    the provided keyword. When 'fuzz' is True, it returns all files that contain the keyword;
    otherwise, it returns only the files whose name exactly matches the keyword.

    Args:
        part_file_name (str): The substring or keyword to search for in file names.
        directory (str): The directory path to search in.
        fuzz (bool, optional): If True, returns all files containing the keyword. If False,
            returns only files with an exact name match. Defaults to True.

    Returns:
        List[str]: A list of file paths that match the search criteria.

    Example:
        >>> # Assuming "/tmp/test" contains files "example.txt" and "sample.txt"
        >>> filter_files("exam", "/tmp/test", fuzz=True)
        ['/tmp/test/example.txt']
    """
    # Note: Using get_file_paths instead of a non-existent fetch_file_paths.
    file_paths = get_file_paths(directory)
    if fuzz:
        return [
            str(file_path)
            for file_path in file_paths
            if part_file_name in os.path.basename(str(file_path))
        ]
    else:
        return [
            str(file_path)
            for file_path in file_paths
            if part_file_name == os.path.basename(str(file_path))
        ]


def list_directories(path: str) -> Optional[List[str]]:
    """
    List all subdirectories in the specified path.

    This function checks whether the given path exists and then returns a list of names for
    each entry in the path that is a directory.

    Args:
        path (str): The path where directories should be listed.

    Returns:
        Optional[List[str]]: A list of directory names if the path exists; otherwise, None.

    Example:
        >>> dirs = list_directories("/tmp")
        >>> isinstance(dirs, list) or dirs is None
        True
    """
    # Ensure the path exists.
    if not os.path.exists(path):
        print("The specified path does not exist.")
        return None

    # List all entries in the path.
    entries = os.listdir(path)
    directories = [
        entry for entry in entries if os.path.isdir(os.path.join(path, entry))
    ]
    return directories


def get_temp_folder() -> str:
    """
    Obtain the path of a new temporary folder.

    This function creates a temporary directory using Python's tempfile module and returns
    its path.

    Returns:
        str: The path to the newly created temporary folder.

    Example:
        >>> temp_folder = get_temp_folder()
        >>> os.path.isdir(temp_folder)
        True
    """
    return tempfile.TemporaryDirectory().name


def get_temp_dir_with_specific_path(*args) -> str:
    """
    Create (if necessary) and return a temporary directory with a specified subpath.

    This function constructs a path within the system's temporary directory using the provided
    arguments. If the directory does not already exist, it is created.

    Args:
        *args: Variable length arguments that specify the subdirectory path components.

    Returns:
        str: The full path of the temporary directory created or existing.

    Example:
        >>> temp_dir = get_temp_dir_with_specific_path("myapp", "cache")
        >>> temp_dir.endswith(os.path.join("myapp", "cache"))
        True
    """
    temp_dir = tempfile.gettempdir()
    flame_temp = os.path.join(temp_dir, *args)
    if not os.path.isdir(flame_temp):
        os.makedirs(flame_temp, exist_ok=True)
    return flame_temp
