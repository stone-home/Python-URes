import bibtexparser
from typing import Union, List, Optional
from pathlib import Path
from .middlewares import (
    FieldNormalizationMiddleware,
    TypeNormalizationMiddleware,
    RuleBasedValidationMiddleware,
)
from .rules import BibRuleRegister


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

    @property
    def rules(self) -> BibRuleRegister:
        return self._rules

    @property
    def bibliograph_library(self) -> bibtexparser.Library:
        return self._bibliography

    @property
    def bibliography_entity(self) -> List[bibtexparser.model.Entry]:
        return self._bibliography.entries

    def append_bibliography(self, bib_file_path: Union[str, Path]) -> None:
        new_bib = self.load_from_file(bib_file_path)
        self._bibliography.add(new_bib.entries)

    def load_from_file(self, file_path: str) -> bibtexparser.Library:
        # Implementation for importing bibliography
        return bibtexparser.parse_file(
            file_path,
            append_middleware=[
                bibtexparser.middlewares.SeparateCoAuthors(),
                bibtexparser.middlewares.SplitNameParts(),
                FieldNormalizationMiddleware(rule_register=self.rules),
                TypeNormalizationMiddleware(),
                RuleBasedValidationMiddleware(rule_register=self.rules),
            ],
        )

    def get_entity(self, key_id: str) -> Union[bibtexparser.model.Entry, None]:
        return self.bibliograph_library.entries_dict.get(key_id, None)

    def display_failed_entities(self):
        blocks = self.bibliograph_library.failed_blocks
        for b in blocks:
            first_line = b.raw.split("\n")[0]
            split_line = first_line.split("{")
            key = split_line[-1] if len(split_line) >= 1 else first_line

            class_name = b.__class__.__name__
            print(f"{key}: Failed Reason: {class_name}")
