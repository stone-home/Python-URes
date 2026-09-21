"""Literature search, citation extraction, and bibliography processing.

Examples:
    >>> from ures.literature import CitationManager
    >>> CitationManager().citations
    []
"""

from .citation import *
from .search import *
from .search_cli import LiteratureSearchCLI


__all__ = [
    "CitationManager",
    "BibManager",
    "BibRuleRegister",
    "BibTypeRule",
    "FormattingRules",
    "OutputRules",
    "StyleConfigError",
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
    "AdapterFactory",
    "QueryParser",
    "Paper",
    "PaperFormatter",
    "LiteratureSearchEngine",
    "DatabaseConfig",
    "LiteratureSearchCLI",
]
