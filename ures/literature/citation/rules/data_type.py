from bibtexparser.middlewares import BlockMiddleware as BibBlockMiddleware
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Type


# proceedings_style = {
# 	"IEEE": "remove",
# 	"ACM": "proceedings",
# 	"VLDB": "proceedings",
# 	"NeurIPS": "minimal",
# 	"ICML": "remove",
# 	"ICLR": "remove",
# 	"AAAI": "minimal",
# 	"IJCAI": "minimal",
# }
@dataclass
class FormattingRules:
    """Formatting rules for bibliography entries."""

    year_range: tuple[int, int] = (1000, 2100)
    page_separator: str = "--"  # e.g., "12--34"
    proceedings_style: str = (
        "full"  # e.g., "full", "short", "proceedings", "minimal", "remove"
    )


@dataclass
class OutputRules:
    """Output formatting rules."""

    field_order: List[str] = field(default_factory=list)
    conditional_fields: Dict[str, str] = field(
        default_factory=dict
    )  # field -> condition
    required_middlewares: List[Type[BibBlockMiddleware]] = field(default_factory=list)
    max_authors: int = field(default=5)


@dataclass
class BibTypeRule:
    """Rules for a specific BibTeX entry type."""

    entry_type: str  # e.g., "article", "inproceedings"
    standard_name: str = "default"
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    forbidden_fields: List[str] = field(default_factory=list)
    field_mappings: Dict[str, str] = field(default_factory=dict)

    # Core rule components
    formatting: FormattingRules = field(default_factory=FormattingRules)
    output: OutputRules = field(default_factory=OutputRules)

    # Metadata
    version: str = "1.0"

    def __post_init__(self):
        """Validate rule consistency after creation."""
        if not self.entry_type:
            raise ValueError("Entry type cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def __hash__(self):
        return hash(self.entry_type)

    def __eq__(self, other):
        return all(
            [
                self.entry_type == other.entry_type,
                self.version == other.version,
            ]
        )

    def __repr__(self):
        return f"BibTypeRule({self.entry_type}[{self.standard_name}], version={self.version})"
