import copy
import re
import logging
import pycountry
from typing import Any, Optional
from bibtexparser.middlewares import BlockMiddleware, NameParts
from bibtexparser.model import Entry, Field
from ures.string import string2date
from .rules import BibRuleRegister

logger = logging.getLogger(__name__)


def field_value_present(entry: Entry, key: str) -> bool:
    field = entry.get(key, None)
    if field is None:
        return False
    return field.value not in (None, "", [])


def field_group_present(entry: Entry, spec: str) -> bool:
    for alternative in spec.split("|"):
        parts = alternative.split("+")
        if all(field_value_present(entry, part) for part in parts):
            return True
    return False


class CitationMiddleware(BlockMiddleware):
    def __init__(self, rule_register: Optional[BibRuleRegister] = None):
        super().__init__()
        self.rule_register = rule_register or BibRuleRegister()


class FieldNormalizationMiddleware(CitationMiddleware):
    """Normalize field names (journaltitle -> journal, etc.)"""

    def transform_entry(self, entry: Entry, *args, **kwargs) -> Entry:
        """Transform entry fields."""
        for field in entry.fields:
            # Handle special cases
            key = field.key
            value = field.value
            if key == "pages":
                # Normalize page ranges
                field.value = self._normalize_pages(value)
            else:
                field.value = value
            # Apply field mapping
            rules = self.rule_register.get_rule(entry.entry_type)
            field_mappings = copy.deepcopy(
                self.rule_register.get_default_field_mapping()
            )
            field_mappings.update(rules.field_mappings)
            new_key = field_mappings.get(field.key, field.key)
            field.key = new_key
        return entry

    def _normalize_pages(self, pages_value: str) -> str:
        """Normalize page ranges to consistent format."""
        if not isinstance(pages_value, str):
            return str(pages_value) if pages_value else ""

        # Replace various dash types with standard double dash
        normalized = pages_value.replace("—", "--").replace("–", "--")

        # Ensure single dash becomes double dash for ranges
        if "-" in normalized and "--" not in normalized:
            normalized = normalized.replace("-", "--")

        return normalized.strip()


class LanguageAsciiNormalizationMiddleware(CitationMiddleware):
    def transform_entry(self, entry: Entry, *args, **kwargs) -> Entry:
        for field in entry.fields:
            key = field.key
            value = field.value
            if key == "langid":
                # Normalize language to ISO 639-1 code
                # field, language, is already processed
                _value = self.normalize_language(value)
                field.value = _value

        return entry

    def normalize_language(self, language_str: str) -> Any:
        """Normalize language string to ISO 639-1 code."""
        special_cases = {
            "pinyin": "zh",
        }
        if language_str in special_cases:
            return special_cases[language_str]
        if not isinstance(language_str, str):
            logger.warning(f"Failed to normalize language '{language_str}'")
            return language_str
        language_str = language_str.strip().lower()
        try:
            lang = pycountry.languages.get(name=language_str)
            if lang and hasattr(lang, "alpha_2"):
                return lang.alpha_2
            # Try searching by common name
            for lang in pycountry.languages:
                if language_str in str(lang.name).lower():
                    if hasattr(lang, "alpha_2"):
                        return lang.alpha_2
        except:
            logger.warning(f"Failed to normalize language '{language_str}'")
            return language_str
        return language_str


class DateSpiltToYearMonthDayMiddleware(CitationMiddleware):
    def transform_entry(self, entry: Entry, *args, **kwargs) -> Entry:
        if entry.get("date", default=None) is not None:
            date_field = entry.pop("date")
            date_parts = string2date(date_field.value)
            # Modify the Year Part
            year_field = copy.deepcopy(date_field)
            year_field.key = "year"
            year_field.value = date_parts["year"]
            # Create Month Parts if available
            month_field = copy.deepcopy(date_field)
            month_field.key = "month"
            month_field.value = str(date_parts["month"]).lower()
            # Create Day Parts if available
            day_field = copy.deepcopy(date_field)
            day_field.key = "day"
            day_field.value = date_parts["day"]

            # Add new fields to entry
            entry.set_field(year_field)
            entry.set_field(month_field)
            entry.set_field(day_field)
        return entry


class PublisherNormalizationMiddleware(CitationMiddleware):
    def transform_entry(self, entry: Entry, *args, **kwargs) -> Entry:
        for field in entry.fields:
            # Handle special cases
            key = field.key
            value = field.value
            if (
                key == "publisher"
                and entry.entry_type in ["inproceedings", "article"]
                and value.lower() in ["{ieee}", "{acm}", "ieee", "acm"]
            ):
                field.value = f"{value} Inc."
        return entry


class ProceedingsNormalizationMiddleware(CitationMiddleware):
    def transform_entry(self, entry: Entry, *args, **kwargs) -> Entry:
        for field in entry.fields:
            # Handle special cases
            key = field.key
            value = field.value
            if key == "booktitle" and entry.entry_type == "inproceedings":
                # Normalize proceedings title
                field.value = self.normailize_proceedings(value)
        return entry

    def normailize_proceedings(self, proceedings_str: str) -> str:
        """Normalize proceedings string to standard format."""
        patterns = [
            r"\bIn\s+Proceedings\s+of\s+the\s+",
            r"\bIn\s+Proceedings\s+of\s+",
            r"\bIn\s+Proc\.\s+of\s+the\s+",
            r"\bIn\s+Proc\.\s+of\s+",
            r"\bProceedings\s+of\s+the\s+",
            r"\bProceedings\s+of\s+",
            r"\bProc\.\s+of\s+the\s+",
            r"\bProc\.\s+of\s+",
            r"\bIn\s+(?=\d|\w+\s+(International|Annual|ACM|IEEE))",
            r"\bProc\.\s+",
        ]
        proceedings_str = proceedings_str.strip()
        for pattern in patterns:
            if re.search(pattern, proceedings_str, re.IGNORECASE):
                proceedings_str = re.sub(
                    pattern, "", proceedings_str, flags=re.IGNORECASE
                )
                proceedings_str = re.sub(r"\s+", " ", proceedings_str).strip()
                break
        prefix_map = {
            "remove": "",
            "full": "In Proceedings of the ",
            "short": "In Proc. of the ",
            "proceedings": "Proceedings of the ",
            "minimal": "In ",
            "proc": "Proc. ",
        }

        style = self.rule_register.get_rule(
            "inproceedings"
        ).formatting.proceedings_style
        prefix = prefix_map.get(style, "full")
        return f"{prefix}{proceedings_str}".strip()


class AcmConferenceVenueMiddleware(CitationMiddleware):
    """Put the conference city on the field ACM actually prints.

    ACM-Reference-Format appends ``location`` or ``city`` to the booktitle
    and prints ``address`` after the publisher. Zotero's BibTeX export has
    only ``address`` for Place, and that Place is the venue city. BibLaTeX
    ``venue`` is the event location.     For ACM and the default style, move the venue city onto ``location``
    when ``location`` and ``city`` are empty. A publisher ``address`` is
    left in place when ``venue`` already holds the city. Books and IEEE
    entries are unchanged.
    """

    def transform_entry(self, entry: Entry, *args, **kwargs) -> Entry:
        if getattr(self.rule_register, "style_name", "") not in {"acm", "default"}:
            return entry
        if entry.entry_type.lower() not in {"inproceedings", "conference"}:
            return entry
        if field_value_present(entry, "location") or field_value_present(entry, "city"):
            return entry
        venue = entry.get("venue", None)
        if venue is not None and venue.value not in (None, "", []):
            venue.key = "location"
            return entry
        address = entry.get("address", None)
        if address is not None and address.value not in (None, "", []):
            address.key = "location"
        return entry


class TypeNormalizationMiddleware(CitationMiddleware):
    """Normalize entry types (conference -> inproceedings, etc.)"""

    def transform_entry(self, entry: Entry, *args, **kwargs) -> Entry:
        new_type = self.rule_register.get_default_bib_type_mapping().get(
            entry.entry_type.lower(), entry.entry_type
        )
        entry.entry_type = new_type
        return entry


class RuleBasedValidationMiddleware(CitationMiddleware):
    """Validate entries against predefined rules."""

    def __init__(self, rule_register: Optional[BibRuleRegister] = None):
        super().__init__()
        self.rule_register = rule_register or BibRuleRegister()

    def transform_entry(self, entry: Entry, *args, **kwargs):
        """Validate entry using dataclass rules."""
        is_valid = True
        rule = self.rule_register.get_rule(entry.entry_type)
        missing_required = []
        for req_field in rule.required_fields:
            if not field_group_present(entry, req_field):
                missing_required.append(req_field)

        if len(missing_required) > 0:
            is_valid = False

        suggested_fields = getattr(rule, "suggested_fields", [])
        if not isinstance(suggested_fields, list):
            suggested_fields = []
        missing_suggested = []
        for spec in suggested_fields:
            if not field_group_present(entry, spec):
                missing_suggested.append(spec)

        is_valid_field = Field(key="is_valid", value=is_valid)
        missing_fields = Field(key="missing_fields", value=missing_required)
        missing_suggested_field = Field(
            key="missing_suggested", value=missing_suggested
        )
        entry.set_field(is_valid_field)
        entry.set_field(missing_fields)
        entry.set_field(missing_suggested_field)

        return entry


class OutputCleanupNoneResultMiddleware(CitationMiddleware):
    """Cleanup entries that are invalid or have missing required fields."""

    def transform_entry(self, entry: Entry, *args, **kwargs) -> Optional[Entry]:
        need_to_removed = []
        for field in entry.fields:
            if field.value in [None, "", [], "none"]:
                need_to_removed.append(field)
        for key in ("is_valid", "missing_fields", "missing_suggested"):
            if entry.get(key, None) is not None:
                entry.pop(key, None)
        for field in need_to_removed:
            entry.pop(field.key, None)
        return entry


class OutputOnlyDesiredFieldsMiddleware(CitationMiddleware):
    """Keep only desired fields in the output."""

    def __init__(self, rule_register: Optional[BibRuleRegister] = None):
        super().__init__()
        self.rule_register = rule_register or BibRuleRegister()

    def transform_entry(self, entry: Entry, *args, **kwargs) -> Optional[Entry]:
        rules = self.rule_register.get_rule(entry.entry_type)

        # remove all fields not in required or optional
        forbidden_fields = rules.forbidden_fields + [
            "is_valid",
            "missing_fields",
            "missing_suggested",
        ]
        for field in forbidden_fields:
            entry.pop(field, None)

        new_fields = []
        allowed_fields = rules.required_fields + rules.optional_fields
        for field in entry.fields:
            if field.key in allowed_fields:
                new_fields.append(field)
        entry.fields = new_fields

        if len(rules.output.required_middlewares) > 0:
            for middleware in rules.output.required_middlewares:
                _middleware_instance = middleware()
                entry = _middleware_instance.transform_entry(entry, *args, **kwargs)
        return entry


class OutputLimitMaxAuthors(CitationMiddleware):
    """Keep only desired fields in the output."""

    def __init__(self, rule_register: Optional[BibRuleRegister] = None):
        super().__init__()
        self.rule_register = rule_register or BibRuleRegister()

    def transform_entry(self, entry: Entry, *args, **kwargs) -> Optional[Entry]:
        rules = self.rule_register.get_rule(entry.entry_type)
        max_authors = rules.output.max_authors
        register_max = getattr(self.rule_register, "max_authors", None)
        if isinstance(register_max, int):
            max_authors = register_max
        if max_authors is None or max_authors <= 0:
            return entry
        for field in entry.fields:
            if field.key == "author" and isinstance(field.value, list):
                is_over_limitation = len(field.value) > max_authors
                max_author_name_parts: list[NameParts] = field.value[:max_authors]
                if is_over_limitation:
                    max_author_name_parts.append(NameParts(first=["others"], last=[]))
                field.value = max_author_name_parts
                break
        return entry
