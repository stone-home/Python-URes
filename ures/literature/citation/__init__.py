import logging
import copy
import bibtexparser
from pathlib import Path
from typing import Optional, Union, List, Type
from .manager import BibManager
from .rules import BibRuleRegister, BibTypeRule, FormattingRules, OutputRules
from .extractors import BBLCitationExtractor, TexCitationExtractor, CitationInfo
from .middlewares import (
    OutputOnlyDesiredFieldsMiddleware,
    OutputCleanupNoneResultMiddleware,
    OutputLimitMaxAuthors,
    TypeNormalizationMiddleware,
    RuleBasedValidationMiddleware,
    FieldNormalizationMiddleware,
    CitationMiddleware,
    LanguageAsciiNormalizationMiddleware,
    DateSpiltToYearMonthDayMiddleware,
    ProceedingsNormalizationMiddleware,
    PublisherNormalizationMiddleware,
)

logger = logging.getLogger(__name__)


class CitationManager:
    def __init__(
        self,
        bibliography_files: Optional[
            Union[Union[str, Path], List[Union[str, Path]]]
        ] = None,
        bibliography_style: str = "default",
    ):
        # Load bibliography files
        if bibliography_files is None:
            bibliography_files = []
        if not isinstance(bibliography_files, list):
            bibliography_files = [bibliography_files]
        self._bib_manager = BibManager(bibliography_style=bibliography_style)
        for bib_file in bibliography_files:
            self._bib_manager.append_bibliography(bib_file)
        # Store citation items for quick access
        self._citations: list[CitationInfo] = []

    @property
    def rules(self) -> BibRuleRegister:
        return self._bib_manager.rules

    @property
    def bib_library(self) -> bibtexparser.Library:
        return self._bib_manager.bibliograph_library

    @property
    def manager(self) -> BibManager:
        return self._bib_manager

    @property
    def citations(self) -> List[CitationInfo]:
        return copy.deepcopy(self._citations)

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

    def _add_citation(self, citation: Union[CitationInfo, List[CitationInfo]]) -> None:
        """Add a citation or list of citations to the manager."""
        if not isinstance(citation, list):
            citation = [citation]
        for cite in citation:
            _bibliography = self._bib_manager.get_entity(cite.key)
            if _bibliography is not None:
                cite.bibliography = _bibliography
            else:
                logger.warning(
                    f"Citation key {cite.key} not found in bibliography items."
                )
            self._citations.append(cite)

    def display_invalid_citations(self) -> None:
        """Display citations that do not have corresponding bibliography entries."""
        for cite in self._citations:
            if (
                cite.bibliography is not None
                and cite.bibliography.get("is_valid", False) is False
            ):
                print(
                    f"Citation Key: {cite.key} is invalid. Missing Fields: {cite.bibliography.get('missing_fields', [])}"
                )

    def to_library(self) -> bibtexparser.Library:
        """Convert all citations to a bibtexparser Library."""
        lib = bibtexparser.Library()
        if len(self.manager.bibliograph_library.strings) > 0:
            lib.add(self.manager.bibliograph_library.strings)
        for cite in self._citations:
            if cite.bibliography is not None:
                lib.add(copy.deepcopy(cite.bibliography))
        return lib

    def save_bibliography(
        self,
        file_path: str,
        middlewares: Optional[
            List[
                Union[
                    Type[bibtexparser.middlewares.BlockMiddleware],
                    bibtexparser.middlewares.BlockMiddleware,
                ]
            ]
        ] = None,
    ) -> None:
        """Save all bibliography entries to a BibTeX file.

        Args:
            file_path (Union[str, Path]): Path to save the BibTeX file.
            middlewares (Optional[List[Type[CitationMiddleware]]]): List of middleware classes to process the entries before saving.
                If None, no additional middleware will be applied. Defaults to None.
        """
        self._bib_manager.export_to_file(
            file_path=file_path, library=self.to_library(), middlewares=middlewares
        )


__all__ = [
    "CitationManager",
    "BibManager",
    "BibRuleRegister",
    "BibTypeRule",
    "FormattingRules",
    "OutputRules",
    "CitationInfo",
    "CitationMiddleware",
    "OutputOnlyDesiredFieldsMiddleware",
    "OutputCleanupNoneResultMiddleware",
    "OutputLimitMaxAuthors",
    "TypeNormalizationMiddleware",
    "RuleBasedValidationMiddleware",
    "FieldNormalizationMiddleware",
    "CitationMiddleware",
    "LanguageAsciiNormalizationMiddleware",
    "DateSpiltToYearMonthDayMiddleware",
    "ProceedingsNormalizationMiddleware",
    "PublisherNormalizationMiddleware",
]
