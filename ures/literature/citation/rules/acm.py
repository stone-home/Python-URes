from typing import List
from .data_type import BibTypeRule
from .basic import BasicRequiredFields, FormattingRules


ACMBibStyle: List[BibTypeRule] = [
    BibTypeRule(
        entry_type="misc",
        standard_name="acm",
        required_fields=BasicRequiredFields,
        optional_fields=[],
        forbidden_fields=[],
    ),
    BibTypeRule(
        entry_type="article",
        standard_name="acm",
        required_fields=BasicRequiredFields
        + ["journal", "volume", "number", "pages", "issn", "url", "address"],
        optional_fields=[],
        forbidden_fields=[],
    ),
    BibTypeRule(
        entry_type="inproceedings",
        standard_name="acm",
        required_fields=BasicRequiredFields
        + ["booktitle", "publisher", "doi", "address", "url", "pages"],
        optional_fields=[],
        forbidden_fields=[],
        formatting=FormattingRules(proceedings_style="proceedings"),
    ),
    BibTypeRule(
        entry_type="book",
        standard_name="acm",
        required_fields=BasicRequiredFields + ["publisher", "isbn", "address"],
        optional_fields=["volume", "edition"],
        forbidden_fields=[],
    ),
    BibTypeRule(
        entry_type="patent",
        standard_name="acm",
        required_fields=BasicRequiredFields + ["note", "url"],
        optional_fields=[],
        forbidden_fields=[],
    ),
    BibTypeRule(
        entry_type="techreport",
        standard_name="acm",
        required_fields=BasicRequiredFields + ["institution", "address", "url"],
        optional_fields=[],
        forbidden_fields=[],
    ),
    BibTypeRule(
        entry_type="thesis",
        standard_name="acm",
        required_fields=BasicRequiredFields + ["publisher", "address", "url"],
        optional_fields=[],
        forbidden_fields=[],
    ),
    BibTypeRule(
        entry_type="preprint",
        standard_name="acm",
        required_fields=BasicRequiredFields + ["doi", "url"],
        optional_fields=["eprint", "archivePrefix"],
        forbidden_fields=[],
    ),
    BibTypeRule(
        entry_type="online",
        standard_name="acm",
        required_fields=BasicRequiredFields + ["url", "urldate"],
        optional_fields=[],
        forbidden_fields=[],
        field_mappings={"titleaddon": "organization"},
    ),
    BibTypeRule(
        entry_type="video",
        standard_name="acm",
        required_fields=BasicRequiredFields,
        optional_fields=["note"],
        forbidden_fields=[],
    ),
    BibTypeRule(
        entry_type="software",
        standard_name="acm",
        required_fields=BasicRequiredFields + ["url", "urldate"],
        optional_fields=["version", "note"],
        forbidden_fields=[],
        field_mappings={"publisher": "organization"},
    ),
    BibTypeRule(
        entry_type="dataset",
        standard_name="acm",
        required_fields=BasicRequiredFields + ["url", "urldate", "publisher"],
        optional_fields=["version", "note"],
        forbidden_fields=[],
    ),
]
