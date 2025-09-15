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
        }

        style = self.rule_register.get_rule(
            "inproceedings"
        ).formatting.proceedings_style
        prefix = prefix_map.get(style, "full")
        return f"{prefix}{proceedings_str}".strip()


class TypeNormalizationMiddleware(CitationMiddleware):
    """Normalize entry types (conference -> inproceedings, etc.)"""

    def transform_entry(self, entry: Entry, *args, **kwargs) -> Entry:
        new_type = self.rule_register.get_defulat_bib_type_mapping().get(
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
            field_value = entry.get(req_field, None)
            if field_value is None:
                missing_required.append(req_field)
            else:
                if field_value.value in (None, "", []):
                    missing_required.append(req_field)

        if len(missing_required) > 0:
            is_valid = False

        is_valid_field = Field(key="is_valid", value=is_valid)
        missing_fields = Field(key="missing_fields", value=missing_required)
        entry.set_field(is_valid_field)
        entry.set_field(missing_fields)

        return entry


class OutputCleanupNoneResultMiddleware(CitationMiddleware):
    """Cleanup entries that are invalid or have missing required fields."""

    def transform_entry(self, entry: Entry, *args, **kwargs) -> Optional[Entry]:
        need_to_removed = []
        for field in entry.fields:
            if field.value in [None, "", [], "none"]:
                need_to_removed.append(field)
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
        forbidden_fields = rules.forbidden_fields + ["is_valid", "missing_fields"]
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
        for field in entry.fields:
            if field.key == "author" and isinstance(field.value, list):
                is_over_limitation = len(field.value) > rules.output.max_authors
                max_author_name_parts: list[NameParts] = field.value[
                    : rules.output.max_authors
                ]
                if is_over_limitation:
                    max_author_name_parts.append(NameParts(first=["others"], last=[]))
                field.value = max_author_name_parts
                break
        return entry
