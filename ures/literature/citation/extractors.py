import bibtexparser
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Optional, List
from dataclasses import dataclass, field


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
    bibliography: Optional[bibtexparser.model.Entry] = None

    def __repr__(self):
        is_valid = False
        if self.bibliography is not None:
            is_valid_field = self.bibliography.get("is_valid", None)
            if is_valid_field is not None:
                is_valid = is_valid_field.value
        return f"[{is_valid}]Cite Key: {self.key}"


class AbcCitationExtractor(ABC):
    @abstractmethod
    def extract_citations(self, tex_file: Union[str, Path]) -> list[CitationInfo]:
        pass


class TexCitationExtractor(AbcCitationExtractor):
    citation_patterns = [
        r"\\cite\{([^}]+)\}",
        r"\\cite\[([^\]]*)\]\{([^}]+)\}",  # with optional arguments
        r"\\citep\{([^}]+)\}",
        r"\\citet\{([^}]+)\}",
        r"\\citeauthor\{([^}]+)\}",
        r"\\citeyear\{([^}]+)\}",
        r"\\citealp\{([^}]+)\}",
        r"\\citealt\{([^}]+)\}",
        r"\\parencite\{([^}]+)\}",
        r"\\textcite\{([^}]+)\}",
        r"\\nocite\{([^}]+)\}",  # Add more patterns as needed
    ]

    def extract_citations(self, tex_file: Union[str, Path]) -> list[CitationInfo]:
        """Parse a .tex file to find all cited bibliography keys.

        Args:
                tex_file (str, Path): Path to the .tex file.

        Returns:
                Set of CitationInfo objects representing cited keys and their locations.
        """
        cited: dict[str, CitationInfo] = {}
        tex_file = Path(tex_file)

        with open(tex_file, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.split("\n")
        clean_lines = []
        for index, line in enumerate(lines):
            # ensure the % is not escaped
            pos = 0
            while True:
                idx = line.find("%", pos)
                if idx == -1:
                    clean_lines.append((line[pos:], index + 1))
                    break
                if idx == 0 or line[idx - 1] != "\\":
                    clean_lines.append((line[:idx], index + 1))
                    break
                pos = idx + 1

        for clean_line, line_no in clean_lines:
            for pattern in self.citation_patterns:
                matches = re.finditer(pattern, clean_line)
                for match in matches:
                    # fetch the last group (the one containing citation keys)
                    keys_str = match.groups()[-1]
                    # split by comma and strip whitespace
                    keys = [k.strip() for k in keys_str.split(",")]
                    for key in keys:
                        if key not in cited:
                            cited[key] = CitationInfo(
                                key=key,
                                sources=[
                                    CitationSource(
                                        source_file=tex_file.name,
                                        line_number=line_no,
                                        source_type="tex",
                                    )
                                ],
                            )
                        else:
                            cited[key].sources.append(
                                CitationSource(
                                    source_file=tex_file.name,
                                    line_number=line_no,
                                    source_type="tex",
                                )
                            )
        return list(cited.values())


class BBLCitationExtractor(AbcCitationExtractor):
    def extract_citations(self, bbl_file: Union[str, Path]) -> List[CitationInfo]:
        """Parse a .bbl file to extract all bibliography entries.

        Args:
                bbl_file (str, Path): Path to the .bbl file.

        Returns:

        """
        cited = []
        try:
            bbl_file = Path(bbl_file)
            with open(bbl_file, "r", encoding="utf-8") as f:
                content = f.read()

                bibitem_pattern = re.compile(
                    r"\\bibitem(?:\[[^\]]*\])?.*?\{(.+?)\}", re.DOTALL
                )
                for match in bibitem_pattern.finditer(content):
                    key = match.group(1)
                    # Find the line number of the match for better context
                    line_num = content.count("\n", 0, match.start()) + 1
                    cited.append(
                        CitationInfo(
                            key=key,
                            sources=[
                                CitationSource(
                                    source_file=bbl_file.name,
                                    line_number=line_num,
                                    source_type="bbl",
                                )
                            ],
                        )
                    )
        except FileNotFoundError:
            pass
        return cited


class AuxCitationExtractor(AbcCitationExtractor):
    """Read cite keys from a BibTeX `.aux` file produced by LaTeX.

    Records come from `\\citation{...}` (including comma-separated keys) and
    nested `\\@input{other.aux}` files. `\\citation{*}` is kept as the key `*`.
    """

    _citation_pattern = re.compile(r"\\citation\{([^}]*)\}")
    _input_pattern = re.compile(r"\\@input\{([^}]*)\}")

    def extract_citations(self, aux_file: Union[str, Path]) -> list[CitationInfo]:
        cited: dict[str, CitationInfo] = {}
        self._extract_from(Path(aux_file), cited, visited=set())
        return list(cited.values())

    def _extract_from(
        self,
        aux_file: Path,
        cited: dict[str, CitationInfo],
        visited: set[Path],
    ) -> None:
        resolved = aux_file.resolve()
        if resolved in visited:
            return
        if not aux_file.is_file():
            raise FileNotFoundError(str(aux_file))
        visited.add(resolved)

        content = aux_file.read_text(encoding="utf-8")
        for match in self._citation_pattern.finditer(content):
            line_no = content.count("\n", 0, match.start()) + 1
            for key in (part.strip() for part in match.group(1).split(",")):
                if not key:
                    continue
                source = CitationSource(
                    source_file=aux_file.name,
                    line_number=line_no,
                    source_type="aux",
                )
                if key not in cited:
                    cited[key] = CitationInfo(key=key, sources=[source])
                else:
                    cited[key].sources.append(source)

        for match in self._input_pattern.finditer(content):
            nested_name = match.group(1).strip()
            if not nested_name:
                continue
            nested = Path(nested_name)
            if not nested.is_absolute():
                nested = aux_file.parent / nested_name
            self._extract_from(nested, cited, visited)
