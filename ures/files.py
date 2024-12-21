import os
import tempfile
from typing import List, Optional


def get_file_paths(directory) -> List[str]:
    """Get all file paths in the specified directory.

    Args:
        directory (str): the directory path

    Returns:
        List[str]: a list of file paths

    """
    file_paths = []
    # Walk through the directory and its subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            # Join the root path with the file name to get the absolute path
            file_path = os.path.join(root, file)
            file_paths.append(str(file_path))

    return file_paths


def filter_files(part_file_name: str, directory: str, fuzz: bool = True) -> List[str]:
    """Fetch all file path that contains the specified part of file name.

    Args:
        part_file_name (str): the part of file name or keyword
        directory (str): the directory path
        fuzz (bool): if True, return all files that contain the part of file name. Defaults to True.

    Returns:
        List[str]: a list of file paths
    """
    file_paths = fetch_file_paths(directory)
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
    """list all directories in the specified path.

    Args:
        path (str): the path to a directories

    Returns:
        Optional[List[str]]: a list of directories or None if the path does not exist

    """
    # Ensure the path exists
    if not os.path.exists(path):
        print("The specified path does not exist.")
        return None

    # List all entries in the path
    entries = os.listdir(path)
    directories = [
        entry for entry in entries if os.path.isdir(os.path.join(path, entry))
    ]

    return directories


def get_temp_folder() -> str:
    """Get a temporary folder.

    Returns:
        str: a temporary folder path

    """
    return tempfile.TemporaryDirectory().name


def get_temp_dir_with_specific_path(*args):
    """Get a temporary directory with a specific path.

    Args:
        *args: Variable length argument list to specify the subdirectory path.

    Returns:
        str: The path of the created temporary directory.
    """
    temp_dir = tempfile.gettempdir()
    flame_temp = os.path.join(temp_dir, *args)
    if not os.path.isdir(flame_temp):
        os.makedirs(flame_temp, exist_ok=True)
    return flame_temp
