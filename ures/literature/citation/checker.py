import logging
import unidecode
import bibtexparser
from bibtexparser.model import Field as BibField
from typing import Union, List, Type, Dict, Any
from pathlib import Path
from ures.string import string2date
from .extractors import BBLCitationExtractor, TexCitationExtractor
from .data_types import (
    CitationInfo,
    BibBaseEntry,
    PatentBib,
    ArticleBib,
    BookBib,
    InproceedingsBib,
    TechReportBib,
    ThesisBib,
    PreprintBib,
    OnlineDocumentBib,
    VideoBib,
    SoftwareBib,
    DatasetBib,
)

logger = logging.getLogger(__name__)


class BibtexFactory:
    type_map: Dict[str, Type[BibBaseEntry]] = {
        "article": ArticleBib,
        "book": BookBib,
        "inproceedings": InproceedingsBib,
        "techreport": TechReportBib,
        "thesis": ThesisBib,
        "phdthesis": ThesisBib,
        "mastersthesis": ThesisBib,
        "preprint": PreprintBib,
        "online": OnlineDocumentBib,
        "video": VideoBib,
        "software": SoftwareBib,
        "dataset": DatasetBib,
        "patent": PatentBib,
    }

    field_name_map: Dict[str, str] = {
        "rights": "copyright",
        "location": "address",
        "journaltitle": "journal",
        "titleaddon": "journal",
        # "date": "year", # A special case, handled separately
    }

    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> List[BibBaseEntry]:
        file_path = Path(file_path)
        if not file_path.is_file():
            logger.error(f"File {file_path} does not exist or is not a file.")
            return []
        bib_database = bibtexparser.parse_file(
            str(file_path),
            append_middleware=[
                bibtexparser.middlewares.SeparateCoAuthors(),
                bibtexparser.middlewares.SplitNameParts(),
            ],
        )
        entries = []
        for entry in bib_database.entries:
            try:
                bib_entry = cls.create_entry(
                    entry.entry_type, entry.key, entry.fields_dict
                )
                bib_entry.raw = entry.raw
                bib_entry.validate()
                entries.append(bib_entry)
            except Exception as e:
                logger.error(
                    f"Error creating BibTeX entry for {entry.key} of type {entry.entry_type}: {e}"
                )
        return entries

    @classmethod
    def create_entry(
        cls, bib_type: str, bib_key: str, fields: Dict[str, Any]
    ) -> BibBaseEntry:
        entry_class = cls.type_map.get(bib_type, BibBaseEntry)
        mandatory_fields = {
            "key": bib_key,
            "bib_type": bib_type,
        }
        extra_fields = {}
        for key, value in fields.items():
            key = cls.field_name_map.get(key, key)
            if isinstance(value, BibField):
                # Extract actual value from BibTexParser Field object
                value = value.value
            if isinstance(value, str):
                # Normalize to ASCII
                value = unidecode.unidecode(value)
            if hasattr(entry_class, key):
                mandatory_fields[key] = value
            else:
                if key == "date":
                    date_dict = string2date(value)
                    mandatory_fields["year"] = date_dict["year"]
                    mandatory_fields["month"] = date_dict["month"]
                    mandatory_fields["day"] = date_dict["day"]
                else:
                    # Store in extra fields
                    extra_fields[key] = value
        return entry_class(extra_fields=extra_fields, **mandatory_fields)


class CitationChecker:
    def __init__(
        self, bibliography_files: Union[Union[str, Path], List[Union[str, Path]]]
    ):
        if not isinstance(bibliography_files, list):
            bibliography_files = [bibliography_files]
        self._bibliography_items = self._import_bibliography_items(bibliography_files)
        self._citations: list[CitationInfo] = []

    @property
    def citations(self) -> list[CitationInfo]:
        return self._citations

    def _import_bibliography_items(
        self, files: List[Union[str, Path]]
    ) -> Dict[str, Type[BibBaseEntry]]:
        bib_items = {}
        for bib_file in files:
            bib_entries = BibtexFactory.load_from_file(bib_file)
            for entry in bib_entries:
                if entry.key not in bib_items:
                    bib_items[entry.key] = entry
                else:
                    logger.warning(
                        f"Duplicate bibliography key found: {entry.key} in file {bib_file}. Skipping."
                    )
        return bib_items

    def _add_citation(self, citation: Union[CitationInfo, List[CitationInfo]]) -> None:
        if not isinstance(citation, list):
            citation = [citation]
        for cite in citation:
            is_bibliography = cite.key in self._bibliography_items
            if is_bibliography:
                cite.bibliography = self._bibliography_items[cite.key]
            else:
                logger.warning(
                    f"Citation key {cite.key} not found in bibliography items."
                )
            self._citations.append(cite)

    def import_citations(
        self, files: List[Union[str, Path]], cleanup: bool = False
    ) -> dict[str, CitationInfo]:
        """Import citations from the given files.

        Args:
                        files (List[Union[str, Path]]): List of file paths to import citations from.
                        cleanup (bool): Whether to clean up the citations previously stored. Defaults to False.

        Returns:
                        List[CitationInfo]: List of imported citations.

        """
        unique_citations = {}
        for file in files:
            file_path = Path(file)
            if not file_path.exists():
                continue

            if file_path.suffix == ".tex":
                extractor = TexCitationExtractor()
            elif file_path.suffix == ".bbl":
                extractor = BBLCitationExtractor()
            else:
                logger.warning(
                    f"Unsupported file type: {file_path.suffix} for file {file_path}. Skipping."
                )
                continue

            for cite in extractor.extract_citations(file_path):
                if cite.key not in unique_citations:
                    unique_citations[cite.key] = cite
                else:
                    unique_citations[cite.key].sources.extend(cite.sources)

        # Merge citation information with existing citations
        if cleanup:
            self._citations.clear()
            self._add_citation(list(unique_citations.values()))
        else:
            for key, cite in unique_citations.items():
                is_found = False
                for stored_cite in self._citations:
                    if stored_cite.key == key:
                        stored_cite.sources.extend(cite.sources)
                        is_found = True
                        break
                if not is_found:
                    self._add_citation(cite)

        return unique_citations

    def get_invalid_citations(self) -> List[str]:
        """Get list of citation keys with invalid or missing bibliography.

        Returns:
                        List of citation keys that are invalid or missing
        """
        invalid_citations = []
        for cite in self._citations:
            if cite.bibliography is None:
                invalid_citations.append(cite.key)
            if cite.bibliography is not None and cite.bibliography.is_valid is False:
                invalid_citations.append(cite.key)
        return invalid_citations

    def to_library(self) -> bibtexparser.Library:
        """Convert all bibliography entries to a BibDatabase object.

        Returns:
                        BibDatabase object containing all bibliography entries
        """
        bib_db = bibtexparser.Library()
        blocks = []
        for cite in self.citations:
            entity = cite.to_bibtex_entry()
            if entity is None:
                continue
            blocks.append(entity)

        bib_db.add(blocks)
        return bib_db

    def save_bib(
        self, file_path: Union[str, Path], enable_name_parts_middleware: bool = True
    ) -> None:
        """Save all bibliography entries to a BibTeX file.

        Args:
                        file_path (Union[str, Path]): Path to save the BibTeX file.
                        enable_name_part_middleware (bool): Whether to enable name part middleware. Defaults to True.
        """
        bib_db = self.to_library()
        middlewares = []
        if enable_name_parts_middleware:
            middlewares.append(bibtexparser.middlewares.MergeNameParts())
            middlewares.append(bibtexparser.middlewares.MergeCoAuthors())
        bibtexparser.write_file(file_path, bib_db, append_middleware=middlewares)
