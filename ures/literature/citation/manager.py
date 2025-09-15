import copy
import inspect
import logging
import bibtexparser
from typing import Union, List, Optional, Type
from pathlib import Path
from .middlewares import (
    CitationMiddleware,
    FieldNormalizationMiddleware,
    TypeNormalizationMiddleware,
    RuleBasedValidationMiddleware,
    ProceedingsNormalizationMiddleware,
    PublisherNormalizationMiddleware,
    DateSpiltToYearMonthDayMiddleware,
    LanguageAsciiNormalizationMiddleware,
    OutputOnlyDesiredFieldsMiddleware,
    OutputCleanupNoneResultMiddleware,
    OutputLimitMaxAuthors,
)
from .rules import BibRuleRegister

logger = logging.getLogger(__name__)


class BibManager:
    def __init__(
        self,
        bib_file_path: Optional[Union[str, Path]] = None,
        bibliography_style: str = "default",
    ):
        self._rules = BibRuleRegister(style=bibliography_style)
        self._bibliography: bibtexparser.Library = (
            self.load_from_file(bib_file_path)
            if bib_file_path is not None
            else bibtexparser.Library()
        )
        self._failed_blocks: List[bibtexparser.model.Block] = copy.deepcopy(
            self._bibliography.failed_blocks
        )

    @property
    def rules(self) -> BibRuleRegister:
        return self._rules

    @property
    def bibliograph_library(self) -> bibtexparser.Library:
        return copy.deepcopy(self._bibliography)

    @property
    def bibliography_entity(self) -> List[bibtexparser.model.Entry]:
        return copy.deepcopy(self._bibliography.entries)

    @property
    def failed_blocks(self) -> List[bibtexparser.model.Block]:
        return copy.deepcopy(self._failed_blocks)

    def append_bibliography(
        self,
        bib_file_path: Union[str, Path],
        middlewares: Optional[List[Type[CitationMiddleware]]] = None,
    ) -> None:
        new_bib = self.load_from_file(bib_file_path, middlewares=middlewares)
        self._bibliography.add(new_bib.entries)
        self._failed_blocks.extend(copy.deepcopy(new_bib.failed_blocks))

    def _default_middlewares(self) -> List[Type[CitationMiddleware]]:
        return [
            LanguageAsciiNormalizationMiddleware,
            DateSpiltToYearMonthDayMiddleware,
            ProceedingsNormalizationMiddleware,
            PublisherNormalizationMiddleware,
            FieldNormalizationMiddleware,
            TypeNormalizationMiddleware,
            RuleBasedValidationMiddleware,
        ]

    def load_from_file(
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
    ) -> bibtexparser.Library:
        # Implementation for importing bibliography
        _middlewares: List[bibtexparser.middlewares.BlockMiddleware] = [
            bibtexparser.middlewares.SeparateCoAuthors(),
            bibtexparser.middlewares.SplitNameParts(),
        ]

        for m in middlewares or self._default_middlewares():
            if inspect.isclass(m) and issubclass(m, CitationMiddleware):
                _middlewares.append(m(rule_register=self.rules))
            elif inspect.isclass(m) and issubclass(
                m, bibtexparser.middlewares.BlockMiddleware
            ):
                _middlewares.append(m())
            elif isinstance(m, bibtexparser.middlewares.BlockMiddleware):
                _middlewares.append(m)  # here m is an instance
            else:
                logger.warning(f"Unknown middleware type: {type(m)}, skipping.")

        return bibtexparser.parse_file(
            file_path,
            append_middleware=_middlewares,
        )

    def export_to_file(
        self,
        file_path: str,
        library: bibtexparser.Library,
        middlewares: Optional[
            List[
                Union[
                    Type[bibtexparser.middlewares.BlockMiddleware],
                    bibtexparser.middlewares.BlockMiddleware,
                ]
            ]
        ] = None,
    ) -> None:
        _middlewares: List[bibtexparser.middlewares.BlockMiddleware] = [
            OutputOnlyDesiredFieldsMiddleware(rule_register=self.rules),
            OutputLimitMaxAuthors(rule_register=self.rules),
        ]
        for m in middlewares or []:
            if inspect.isclass(m) and issubclass(m, CitationMiddleware):
                _middlewares.append(m(rule_register=self.rules))
            elif inspect.isclass(m) and issubclass(
                m, bibtexparser.middlewares.BlockMiddleware
            ):
                _middlewares.append(m())
            elif isinstance(m, bibtexparser.middlewares.BlockMiddleware):
                _middlewares.append(m)  # here m is an instance
            else:
                logger.warning(f"Unknown middleware type: {type(m)}, skipping.")
        _middlewares.extend(
            [
                OutputCleanupNoneResultMiddleware(rule_register=self.rules),
                bibtexparser.middlewares.MergeNameParts(),
                bibtexparser.middlewares.MergeCoAuthors(),
                bibtexparser.middlewares.SortFieldsAlphabeticallyMiddleware(),
                bibtexparser.middlewares.SortBlocksByTypeAndKeyMiddleware(),
            ]
        )

        bibtexparser.write_file(file_path, library, append_middleware=_middlewares)

    def get_entity(self, key_id: str) -> Union[bibtexparser.model.Entry, None]:
        return self.bibliograph_library.entries_dict.get(key_id, None)

    def display_failed_entities(self):
        blocks = self.failed_blocks
        for b in blocks:
            first_line = b.raw.split("\n")[0]
            split_line = first_line.split("{")
            key = split_line[-1] if len(split_line) >= 1 else first_line

            class_name = b.__class__.__name__
            print(f"{key}: Failed Reason: {class_name}")
