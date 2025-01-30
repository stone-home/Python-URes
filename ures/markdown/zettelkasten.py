import os
from typing import Optional
from ures.timedate import time_now
from ures.string import zettelkasten_id
from .manipulator import MarkdownDocument, frontmatter


class Zettelkasten(MarkdownDocument):
    """
    Class for handling Zettelkasten markdown files. In my case, a list of mandatory fields is defined in a class
    variable.

    In Zettelkasten note-taking system, only three types of notes are supported: 'fleeting', 'literature', 'permanent',
    but I added 'atom' type for my own use.
    """

    MANDATORY_FIELDS = ["title", "type", "url", "create", "id", "tags", "aliases"]
    ALLOWED_TYPES = ["fleeting", "literature", "permanent", "atom"]

    def __init__(
        self,
        title: str,
        n_type: str,
        url: Optional[str] = None,
        tags: Optional[list] = None,
        aliases: Optional[list] = None,
    ):
        """
        Initialize a Zettelkasten object
        Args:
            title (str): The title of the note.
            n_type (str): The type of the note, only support 'fleeting', 'literature', 'permanent' and 'atom'.
            url (str): The url of the note.
            tags (list): The tags of the note.
            aliases
        """
        if not isinstance(title, str) or not title.strip():
            raise ValueError("Title must be a non-empty string.")
        if n_type not in self.ALLOWED_TYPES:
            raise ValueError(
                f"Invalid type '{n_type}'. Allowed types are: {', '.join(self.ALLOWED_TYPES)}."
            )
        if url is not None and not isinstance(url, str):
            raise ValueError("URL must be a string.")
        if tags is not None and not isinstance(tags, list):
            raise ValueError("Tags must be a list.")
        if aliases is not None and not isinstance(aliases, list):
            raise ValueError("Aliases must be a list.")

        _metadata = {
            "id": zettelkasten_id(),
            "create": time_now(),
        }
        super().__init__(metadata=_metadata)
        self.title = title
        self.type = n_type
        self.url = url or ""
        self.tags = tags or []
        self.aliases = aliases or []

    @classmethod
    def from_file(cls, file_path: str) -> "MarkdownDocument":
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
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        with open(file_path, "r", encoding="utf-8") as f:
            post = frontmatter.load(f)
        zk = cls(
            title=post.metadata.get("title", None),
            n_type=post.metadata.get("type", None),
            url=post.metadata.get("url", None),
            tags=post.metadata.get("tags", None),
            aliases=post.metadata.get("aliases", None),
        )
        zk.metadata["id"] = post.metadata.get("id", zk.metadata["id"])
        zk.metadata["create"] = post.metadata.get("create", zk.metadata["create"])
        zk.add_content(post.content)
        return zk

    @property
    def title(self) -> str:
        return self.get_frontmatter("title")

    @title.setter
    def title(self, value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Title must be a non-empty string.")
        self.set_frontmatter("title", value)

    @property
    def type(self) -> str:
        return self.get_frontmatter("type")

    @type.setter
    def type(self, value):
        if value not in self.ALLOWED_TYPES:
            raise ValueError(
                f"Invalid type '{value}'. Allowed types are: {', '.join(self.ALLOWED_TYPES)}."
            )
        self.set_frontmatter("type", value)

    @property
    def url(self) -> str:
        return self.get_frontmatter("url")

    @url.setter
    def url(self, value):
        if not isinstance(value, str):
            raise ValueError("URL must be a string.")
        self.set_frontmatter("url", value)

    @property
    def tags(self) -> list:
        return self.get_frontmatter("tags")

    @tags.setter
    def tags(self, value):
        if not isinstance(value, list):
            raise ValueError("Tags must be a list.")
        self.set_frontmatter("tags", value)

    @property
    def aliases(self) -> list:
        return self.get_frontmatter("aliases")

    @aliases.setter
    def aliases(self, value):
        if not isinstance(value, list):
            raise ValueError("Aliases must be a list.")
        self.set_frontmatter("aliases", value)

    def add_tag(self, tag: str):
        self.tags.append(tag)

    def remove_tag(self, tag: str):
        self.tags.remove(tag)

    def add_alias(self, alias: str):
        self.aliases.append(alias)

    def remove_alias(self, alias: str):
        self.aliases.remove(alias)
