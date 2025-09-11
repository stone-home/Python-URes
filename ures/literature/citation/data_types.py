import logging
import bibtexparser
from bibtexparser.middlewares.names import NameParts
from pathlib import Path
from typing import Union, Optional, List, Type
from dataclasses import dataclass, field, fields

logger = logging.getLogger(__name__)


# ================= BibTeX Entry Types ================= #
@dataclass(slots=True)
class BibBaseEntry:
    key: str
    bib_type: str
    author: Optional[Union[list[NameParts], str]] = None
    title: str = field(default="")
    year: Optional[int] = None
    month: Optional[str] = None
    day: Optional[int] = None
    is_valid: bool = field(init=False, default=False)
    extra_fields: dict = field(default_factory=dict)
    raw: Optional[str] = None
    _required_fields = ["key", "title", "year", "bib_type", "author"]
    _not_exported_fields = ["is_valid", "raw", "extra_fields", "key", "bib_type", "day"]

    def validate(self) -> bool:
        missing_required = []
        for required_field in self._required_fields:
            field_value = getattr(self, required_field, None)
            if field_value in (None, "", []):
                missing_required.append(required_field)

        if len(missing_required) > 0:
            self.is_valid = False
            logger.warning(
                f"Entry '{self.key}' ({self.bib_type}) is missing required fields: {', '.join(missing_required)}"
            )
        else:
            self.is_valid = True

        return self.is_valid

    def to_entry(self, include_extra: bool = False) -> bibtexparser.model.Entry:
        """Convert dataclass instance to bibtexparser Entry object."""
        self.validate()

        entry_fields = []
        # Get all dataclass fields
        for dc_field in fields(self):
            field_name = dc_field.name
            field_value = getattr(self, field_name)

            # Skip special fields
            if field_name in self._not_exported_fields:
                continue

            # Skip fields starting with underscore (like _required_fields)
            if field_name.startswith("_"):
                continue

            # Skip empty values
            if field_value in (None, "", []):
                continue

            # Handle different value types
            entry_fields.append(bibtexparser.model.Field(field_name, field_value))

        # Add fields from extra_fields
        if include_extra:
            for key, value in self.extra_fields.items():
                if value not in (None, "", []):
                    entry_fields.append(bibtexparser.model.Field(key, value))

        return bibtexparser.model.Entry(self.bib_type, self.key, entry_fields)

    def to_library(self) -> bibtexparser.Library:
        """Create a Library containing only this entry."""
        lib = bibtexparser.Library()
        lib.add(self.to_entry())
        return lib

    def write_to_file(self, filename: str, append: bool = False):
        """Write entry directly to file."""
        lib = self.to_library()

        if append and Path(filename).is_file():
            # If append mode, read existing file first
            existing_lib = bibtexparser.parse_file(filename)
            existing_lib.add(self.to_entry())
            lib = existing_lib

        middlewares = []
        if isinstance(self.author, list) and all(
            isinstance(a, NameParts) for a in self.author
        ):
            middlewares.append(bibtexparser.middlewares.names.MergeNameParts())
            middlewares.append(bibtexparser.middlewares.names.MergeCoAuthors())
        bibtexparser.write_file(filename, lib, append_middleware=middlewares)


@dataclass(slots=True)
class PatentBib(BibBaseEntry):
    note: Optional[str] = None
    url: Optional[str] = None
    _required_fields = [
        "key",
        "title",
        "year",
        "bib_type",
        "author",
        "note",
        "url",
    ]


# ================= Formal Published Types ================= #
@dataclass(slots=True)
class PublishedBibEntry(BibBaseEntry):
    editor: Optional[List[str]] = None
    publisher: Optional[str] = None
    address: Optional[str] = None
    url: Optional[str] = None
    doi: Optional[str] = None
    _required_fields = [
        "key",
        "title",
        "year",
        "bib_type",
        "author",
        "publisher",
        "url",
        "doi",
    ]


@dataclass(slots=True)
class ArticleBib(PublishedBibEntry):
    """BibTeX article entry."""

    journal: Optional[str] = None
    volume: Optional[int] = None
    number: Optional[int] = None
    pages: Optional[str] = None
    issn: Optional[str] = None
    _required_fields = [
        "key",
        "title",
        "year",
        "bib_type",
        "author",
        "journal",
        "volume",
        "number",  # issue number
        "pages",
        "issn",
        "url",
        "address",
    ]


@dataclass(slots=True)
class BookBib(PublishedBibEntry):
    """BibTeX book entry."""

    isbn: Optional[str] = None
    volume: Optional[int] = None
    edition: Optional[str] = None
    _required_fields = [
        "key",
        "title",
        "year",
        "bib_type",
        "author",
        "publisher",
        "isbn",
        "address" "publisher",
    ]


@dataclass(slots=True)
class InproceedingsBib(PublishedBibEntry):
    """BibTeX Conference paper entry."""

    booktitle: Optional[str] = None
    location: Optional[str] = None
    pages: Optional[str] = None
    _required_fields = [
        "key",
        "title",
        "year",
        "bib_type",
        "author",
        "booktitle",
        "publisher",
        "doi",
        "url",
        "address",
    ]


@dataclass(slots=True)
class TechReportBib(PublishedBibEntry):
    source: Optional[str] = None
    _required_fields = [
        "key",
        "title",
        "year",
        "bib_type",
        "author",
        "url",
        "publisher",
        "address",
    ]


@dataclass(slots=True)
class ThesisBib(PublishedBibEntry):
    advisor: Optional[str] = None
    institution: Optional[str] = None
    note: Optional[str] = None
    _required_fields = [
        "key",
        "title",
        "year",
        "bib_type",
        "author",
        "publisher",
        "address",
        "url",
    ]


# ================= Informal Published Types ================= #


@dataclass(slots=True)
class PreprintBib(BibBaseEntry):
    url: Optional[str] = None
    eprint: str = field(default="")
    archivePrefix: str = field(default="arXiv")
    doi: Optional[str] = None
    _required_fields = ["key", "title", "year", "bib_type", "author", "doi", "url"]


@dataclass(slots=True)
class OnlineDocumentBib(BibBaseEntry):
    url: Optional[str] = None
    urldate: Optional[str] = (None,)
    journal: Optional[str] = None
    _required_fields = [
        "key",
        "title",
        "year",
        "bib_type",
        "author",
        "url",
        "urldate",
        "journal",
    ]


@dataclass(slots=True)
class VideoBib(BibBaseEntry):
    note: Optional[str] = None
    _required_fields = ["key", "title", "year", "bib_type", "author"]


@dataclass(slots=True)
class SoftwareBib(BibBaseEntry):
    url: Optional[str] = None
    publisher: Optional[str] = None
    version: Optional[str] = None
    urldate: Optional[str] = None
    _required_fields = [
        "key",
        "title",
        "year",
        "bib_type",
        "author",
        "url",
        "urldate",
        "publisher",
    ]


@dataclass(slots=True)
class DatasetBib(BibBaseEntry):
    url: Optional[str] = None
    publisher: Optional[str] = None
    version: Optional[str] = None
    urldate: Optional[str] = None
    _required_fields = [
        "key",
        "title",
        "year",
        "bib_type",
        "author",
        "url",
        "urldate",
        "publisher",
    ]


# ================= Citation Info =================


@dataclass(frozen=True, slots=True)
class CitationSource:
    """The data structure to hold citation source information."""

    source_file: str
    line_number: int
    source_type: str


@dataclass(slots=True)
class CitationInfo:
    """The data structure to hold citation information."""

    key: str
    sources: List[CitationSource] = field(default_factory=list)
    bibliography: Optional[BibBaseEntry] = None

    def __repr__(self):
        return f"Cite Key: {self.key}"

    def to_bibtex_entry(self) -> Optional[bibtexparser.model.Entry]:
        if self.bibliography:
            return self.bibliography.to_entry()
        return None
