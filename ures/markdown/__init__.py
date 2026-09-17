"""Markdown documents and Zettelkasten notes with YAML front matter.

Examples:
    >>> from ures.markdown import MarkdownDocument
    >>> doc = MarkdownDocument(content="Hello", metadata={"title": "Note"})
    >>> "title" in doc.metadata
    True
"""
from .manipulator import MarkdownDocument, Content, ContentSection
from .zettelkasten import Zettelkasten

__all__ = ["MarkdownDocument", "Zettelkasten", "Content", "ContentSection"]
