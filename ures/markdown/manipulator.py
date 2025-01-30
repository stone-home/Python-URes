import os
import frontmatter
from pathlib import Path
from typing import Any, Dict, Optional, List, AnyStr, Union
from copy import deepcopy


class MarkdownDocument:
    """
    A low-level class for manipulating Markdown files with front matter.

    This class provides methods to add and modify Markdown content and front matter,
    supporting nested structures in front matter (e.g., dictionaries within YAML front matter).
    """

    MANDATORY_FIELDS: List[AnyStr] = []

    def __init__(self, content: str = "", metadata: Optional[Dict[str, Any]] = None):
        """
        Initializes a new MarkdownDocument instance.

        Args:
            content (str): The Markdown content. Defaults to an empty string.
            metadata (Optional[Dict[str, Any]]): The front matter metadata as a dictionary. Defaults to None.

        Example:
            >>> doc = MarkdownDocument(
            ...     content="# Hello World",
            ...     metadata={"title": "Greeting", "tags": ["intro", "welcome"]}
            ... )
        """
        if metadata is None:
            metadata = {}
        self.post = frontmatter.Post(content, **metadata)

    @staticmethod
    def path_preprocess(input_path: Union[str, Path]) -> Path:
        if isinstance(input_path, str):
            output_path = Path(input_path)
        else:
            output_path = input_path
        return output_path

    @classmethod
    def from_file(cls, file_path: Union[Path, str]) -> "MarkdownDocument":
        """
        Creates a MarkdownDocument instance by loading a Markdown file.

        Args:
            file_path (str): The path to the Markdown file.

        Returns:
            MarkdownDocument: An instance representing the loaded Markdown file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            frontmatter.InvalidFrontMatterError: If the front matter is malformed.
        """
        file_path = cls.path_preprocess(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        with open(file_path, "r", encoding="utf-8") as f:
            post = frontmatter.load(f)
        return cls(content=post.content, metadata=deepcopy(post.metadata))

    @property
    def content(self) -> str:
        """
        Retrieves the Markdown content.

        Returns:
            str: The Markdown content.

        Example:
            >>> doc = MarkdownDocument(content="# Hello World")
            >>> doc.content
            "# Hello World"
        """
        return self.post.content

    @content.setter
    def content(self, new_content: str) -> None:
        """
        Sets the Markdown content.

        Args:
            new_content (str): The new Markdown content.

        Example:
            >>> doc = MarkdownDocument()
            >>> doc.content = "# New Title"
        """
        self.post.content = new_content

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Retrieves the front matter metadata.

        Returns:
            Dict[str, Any]: The metadata dictionary.

        Example:
            >>> doc = MarkdownDocument(metadata={"title": "Greeting", "tags": ["intro", "welcome"]})
            >>> doc.metadata
            {"title": "Greeting", "tags": ["intro", "welcome"]}
        """
        return self.post.metadata

    @metadata.setter
    def metadata(self, new_metadata: Dict[str, Any]) -> None:
        """
        Sets the front matter metadata.

        Args:
            new_metadata (Dict[str, Any]): The new metadata dictionary.

        Example:
            >>> doc = MarkdownDocument()
            >>> doc.metadata = {"title": "New Greeting", "tags": ["updated"]}
        """
        self.post.metadata = new_metadata

    def add_content(self, content: str, append: bool = True) -> None:
        """
        Adds content to the Markdown document.

        Args:
            content (str): The Markdown content to add.
            append (bool): If True, appends to existing content; otherwise, prepends.
                           Defaults to True.

        Example:
            >>> doc = MarkdownDocument()
            >>> doc.add_content("# Introduction")
            >>> doc.add_content("Some introductory text.", append=True)
        """
        if append:
            if self.post.content:
                self.post.content += "\n" + content
            else:
                self.post.content = content
        else:
            if self.post.content:
                self.post.content = content + "\n" + self.post.content
            else:
                self.post.content = content

    def set_frontmatter(
        self, key_path: str, value: Any, overwrite: bool = True
    ) -> None:
        """
        Sets a front matter key to a specified value. Supports nested keys using dot notation,
        including mixed types such as dictionaries within lists.

        Args:
            key_path (str): The front matter key path. Use dot notation for nested keys
                            (e.g., "author.name" or "sections.0.title").
            value (Any): The value to set for the key.
            overwrite (bool): If True, overwrites the existing value; otherwise, appends to lists
                              or creates new entries in lists. Defaults to True.

        Example:
            >>> doc = MarkdownDocument()
            >>> doc.set_frontmatter("author.name", "John Doe")
            >>> doc.set_frontmatter("author.contact.email", "john@example.com")
            >>> doc.set_frontmatter("sections.0.title", "Introduction")
            >>> doc.set_frontmatter("sections.0.content", "Welcome to the introduction.")
            >>> doc.set_frontmatter("sections.1.title", "Conclusion")
            >>> doc.set_frontmatter("sections.1.content", "Wrapping up.")
        """
        keys = key_path.split(".")
        current = self.post.metadata

        for i, key in enumerate(keys):
            is_last = i == len(keys) - 1
            # Determine if the current key is meant to be a list index
            if key.isdigit():
                index = int(key)
                if not isinstance(current, list):
                    if overwrite:
                        # Initialize as list
                        parent = self.post.metadata
                        for k in keys[:i]:
                            if k.isdigit():
                                parent = parent[int(k)]
                            else:
                                parent = parent[k]
                        parent[int(keys[i - 1])] = []
                        current = parent[int(keys[i - 1])]
                    else:
                        raise TypeError(
                            f"Expected list at {'.'.join(keys[:i])}, found {type(current).__name__}"
                        )
                # Extend the list if necessary
                while len(current) <= index:
                    current.append({})
                if is_last:
                    if isinstance(current[index], list) and not overwrite:
                        current[index].append(value)
                    elif isinstance(current[index], dict):
                        if isinstance(value, dict):
                            current[index].update(value)
                        else:
                            current[index]["value"] = value
                    elif not overwrite:
                        current[index] = [current[index], value]
                    else:
                        current[index] = value
                else:
                    if not isinstance(current[index], (dict, list)):
                        # Initialize as dict or list based on next key
                        next_key = keys[i + 1]
                        if next_key.isdigit():
                            current[index] = []
                        else:
                            current[index] = {}
                    current = current[index]
            else:
                if not isinstance(current, dict):
                    if overwrite:
                        # Initialize as dict
                        parent = self.post.metadata
                        for k in keys[:i]:
                            if k.isdigit():
                                parent = parent[int(k)]
                            else:
                                parent = parent[k]
                        parent[keys[i - 1]] = {}
                        current = parent[keys[i - 1]]
                    else:
                        raise TypeError(
                            f"Expected dict at {'.'.join(keys[:i])}, found {type(current).__name__}"
                        )
                if is_last:
                    if key in current:
                        if isinstance(current[key], list) and not overwrite:
                            current[key].append(value)
                        elif isinstance(current[key], dict):
                            if isinstance(value, dict):
                                current[key].update(value)
                            else:
                                current[key]["value"] = value
                        elif not overwrite:
                            current[key] = [current[key], value]
                        else:
                            current[key] = value
                    else:
                        current[key] = value
                else:
                    if key not in current or not isinstance(current[key], (dict, list)):
                        # Initialize as dict or list based on next key
                        next_key = keys[i + 1]
                        if next_key.isdigit():
                            current[key] = []
                        else:
                            current[key] = {}
                    current = current[key]

    def get_frontmatter(self, key_path: str) -> Any:
        """
        Retrieves the value of a front matter key. Supports nested keys using dot notation,
        including list indices.

        Args:
            key_path (str): The front matter key path. Use dot notation for nested keys
                            (e.g., "author.name" or "sections.0.title").

        Returns:
            Any: The value associated with the key, or None if the key does not exist.

        Example:
            >>> doc = MarkdownDocument(
            ...     metadata={
            ...         "author": {"name": "John Doe", "contact": {"email": "john@example.com"}},
            ...         "sections": [
            ...             {"title": "Introduction", "content": "Welcome."},
            ...             {"title": "Conclusion", "content": "Goodbye."}
            ...         ]
            ...     }
            ... )
            >>> doc.get_frontmatter("author.name")
            "John Doe"
            >>> doc.get_frontmatter("sections.0.title")
            "Introduction"
            >>> doc.get_frontmatter("sections.1.content")
            "Goodbye."
        """
        keys = key_path.split(".")
        metadata = self.post.metadata

        for key in keys:
            if isinstance(metadata, dict):
                metadata = metadata.get(key, None)
            elif isinstance(metadata, list):
                if key.isdigit():
                    index = int(key)
                    if 0 <= index < len(metadata):
                        metadata = metadata[index]
                    else:
                        return None
                else:
                    return None
            else:
                return None

            if metadata is None:
                return None

        return metadata

    def remove_frontmatter(self, key_path: str) -> None:
        """
        Removes a front matter key. Supports nested keys using dot notation,
        including list indices.

        Args:
            key_path (str): The front matter key path to remove. Use dot notation for nested keys
                            (e.g., "author.contact.email" or "sections.0.title").

        Example:
            >>> doc = MarkdownDocument(
            ...     metadata={
            ...         "author": {"name": "John Doe", "contact": {"email": "john@example.com"}},
            ...         "sections": [
            ...             {"title": "Introduction", "content": "Welcome."},
            ...             {"title": "Conclusion", "content": "Goodbye."}
            ...         ]
            ...     }
            ... )
            >>> doc.remove_frontmatter("author.contact.email")
            >>> doc.get_frontmatter("author.contact.email") is None
            True
            >>> doc.remove_frontmatter("sections.1.title")
            >>> doc.get_frontmatter("sections.1.title") is None
            True
        """
        keys = key_path.split(".")
        metadata = self.post.metadata

        for i, key in enumerate(keys):
            is_last = i == len(keys) - 1
            if isinstance(metadata, dict):
                if key not in metadata:
                    return  # Key path does not exist; nothing to remove
                if is_last:
                    del metadata[key]
                    return
                metadata = metadata[key]
            elif isinstance(metadata, list):
                if key.isdigit():
                    index = int(key)
                    if 0 <= index < len(metadata):
                        if is_last:
                            del metadata[index]
                            return
                        metadata = metadata[index]
                    else:
                        return  # Index out of range; nothing to remove
                else:
                    return  # Invalid key for list; nothing to remove
            else:
                return  # Neither dict nor list; nothing to remove

    def to_markdown(self) -> str:
        """
        Serializes the MarkdownDocument to a Markdown-formatted string, including front matter.

        Returns:
            str: The complete Markdown content with front matter.

        ERROR:
            ValueError: If the front matter is missing mandatory fields.

        Example:
            >>> doc = MarkdownDocument(
            ...     content="# Hello World",
            ...     metadata={"title": "Greeting", "tags": ["intro", "welcome"]}
            ... )
            >>> print(doc.to_markdown())
            ---
            title: Greeting
            tags:
              - intro
              - welcome
            ---

            # Hello World
        """
        self.validation_frontmatter()
        return frontmatter.dumps(self.post)

    def save(self, file_path: Union[Path, str]) -> None:
        """
        Saves the MarkdownDocument to a specified file.

        Args:
            file_path (str): The path where the Markdown file will be saved.

        Example:
            >>> doc = MarkdownDocument(
            ...     content="# Hello World",
            ...     metadata={"title": "Greeting", "tags": ["intro", "welcome"]}
            ... )
            >>> doc.save_to_file("greeting.md")
        """
        file_path = self.path_preprocess(file_path)
        markdown_str = self.to_markdown()
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(markdown_str)

    def load_from_file(self, file_path: Union[Path, str]) -> None:
        """
        Loads Markdown content and front matter from a specified file into the current instance.

        Args:
            file_path (str): The path to the Markdown file to load.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            frontmatter.InvalidFrontMatterError: If the front matter is malformed.

        Example:
            >>> doc = MarkdownDocument()
            >>> doc.load_from_file("existing.md")
        """
        file_path = self.path_preprocess(file_path)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        with open(file_path, "r", encoding="utf-8") as f:
            post = frontmatter.load(f)

        self.post.content = post.content
        self.post.metadata = deepcopy(post.metadata)

    def clear_content(self) -> None:
        """
        Clears all Markdown content, leaving only the front matter.

        Example:
            >>> doc = MarkdownDocument(content="# Hello World")
            >>> doc.clear_content()
            >>> print(doc.content)
            ""
        """
        self.post.content = ""

    def clear_frontmatter(self) -> None:
        """
        Clears all front matter metadata, leaving only the Markdown content.

        Example:
            >>> doc = MarkdownDocument(
            ...     content="# Hello World",
            ...     metadata={"title": "Greeting", "tags": ["intro", "welcome"]}
            ... )
            >>> doc.clear_frontmatter()
            >>> print(doc.metadata)
            {}
        """
        self.post.metadata = {}

    def validation_frontmatter(self):
        """
        Validate the frontmatter metadata against the mandatory fields.
        """
        missing_fields = []
        for field in self.MANDATORY_FIELDS:
            if self.get_frontmatter(field) is None:
                missing_fields.append(field)

        if missing_fields:
            missing = ", ".join(missing_fields)
            raise ValueError(f"Missing mandatory front matter fields: {missing}")
