import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import bibtexparser

from ures.literature.citation.extractors import AuxCitationExtractor
from ures.literature.citation.manager import BibManager
from ures.literature.citation.rules import BibRuleRegister, StyleConfigError
from ures.literature.citation.rules.style_config import (
    LOCAL_STYLE_FILENAME,
    PACKAGED_STYLES,
    PROFILES,
    load_packaged_style,
)

REPORT_FILENAME = "bib-lint-report.json"
FORMATTED_NAME_INFIX = " - formatted"


def _print_error(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)


def _print_warning(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def _format_field_spec(spec: str) -> str:
    return spec.replace("|", " or ").replace("+", " and ")


def _failed_block_key(block: Any) -> str:
    raw = getattr(block, "raw", "") or ""
    first_line = raw.split("\n")[0]
    if "{" in first_line:
        return first_line.split("{")[-1].rstrip(",").strip()
    return first_line or block.__class__.__name__


def _author_count(entry) -> int:
    field = entry.get("author", None)
    if field is None:
        return 0
    value = field.value
    if isinstance(value, list):
        return len(value)
    if isinstance(value, str) and value.strip():
        return len([part for part in value.split(" and ") if part.strip()])
    return 0


def _collect_entry_issues(
    entries, source_name: str, max_authors: Optional[int] = None
) -> tuple[List[Dict[str, Any]], int, int]:
    reports = []
    errors = 0
    warnings = 0
    for entry in entries:
        missing_required_field = entry.get("missing_fields", None)
        missing_required = (
            list(missing_required_field.value or [])
            if missing_required_field is not None
            else []
        )
        missing_suggested_field = entry.get("missing_suggested", None)
        missing_suggested = (
            list(missing_suggested_field.value or [])
            if missing_suggested_field is not None
            else []
        )
        author_count = _author_count(entry)
        severity = "ok"
        if missing_required:
            severity = "error"
            errors += 1
            for spec in missing_required:
                _print_error(
                    f"{source_name}: {entry.key} missing required field {_format_field_spec(spec)}"
                )
        if missing_suggested:
            if severity != "error":
                severity = "warning"
            warnings += len(missing_suggested)
            for spec in missing_suggested:
                _print_warning(
                    f"{source_name}: {entry.key} missing {_format_field_spec(spec)}"
                )
        if (
            isinstance(max_authors, int)
            and max_authors > 0
            and author_count > max_authors
        ):
            if severity != "error":
                severity = "warning"
            warnings += 1
            _print_warning(
                f"{source_name}: {entry.key} has {author_count} authors (max {max_authors})"
            )
        reports.append(
            {
                "key": entry.key,
                "type": entry.entry_type,
                "severity": severity,
                "missing_required": missing_required,
                "missing_suggested": missing_suggested,
                "author_count": author_count,
            }
        )
    return reports, errors, warnings


def cmd_init(style: str) -> int:
    if style not in PACKAGED_STYLES:
        _print_error(f"unknown style '{style}' (use acm or ieee)")
        return 2
    dest = Path.cwd() / LOCAL_STYLE_FILENAME
    if dest.exists():
        _print_error(f"{LOCAL_STYLE_FILENAME} already exists")
        return 2
    data = load_packaged_style(style)
    dest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(str(dest), file=sys.stderr)
    return 0


def formatted_bib_path(bib_path: Path) -> Path:
    return bib_path.with_name(
        f"{bib_path.stem}{FORMATTED_NAME_INFIX}{bib_path.suffix}"
    )


def _cited_keys_from_aux(aux_path: Path) -> Optional[Set[str]]:
    keys = {item.key for item in AuxCitationExtractor().extract_citations(aux_path)}
    if "*" in keys:
        return None
    return keys


def _subset_library(library, keys: Optional[Set[str]]):
    if keys is None:
        return library
    subset = bibtexparser.Library()
    for string in library.strings:
        subset.add(copy.deepcopy(string))
    for entry in library.entries:
        if entry.key in keys:
            subset.add(copy.deepcopy(entry))
    return subset


def _load_manager(bib_path: Path, profile: str) -> BibManager:
    rules = BibRuleRegister.from_json_style(
        style="acm", profile=profile, load_cwd=True
    )
    return BibManager(bib_file_path=bib_path, rules=rules, profile=profile)


def _write_report(payload: Dict[str, Any]) -> None:
    Path.cwd().joinpath(REPORT_FILENAME).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def cmd_process(
    bib_path: Path,
    profile: str,
    write_back: bool,
    aux_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> int:
    if profile not in PROFILES:
        _print_error(
            f"unknown profile '{profile}' (use library, submission, or camera-ready)"
        )
        return 2
    if not bib_path.exists():
        _print_error(f"{bib_path} not found")
        return 2
    if bib_path.suffix != ".bib":
        _print_error(f"{bib_path} is not a .bib file")
        return 2

    dest_path: Optional[Path] = None
    if write_back:
        dest_path = output_path or formatted_bib_path(bib_path)
        if dest_path.suffix != ".bib":
            _print_error(f"{dest_path} is not a .bib file")
            return 2
        if dest_path.resolve() == bib_path.resolve():
            _print_error(f"refusing to overwrite input {bib_path}")
            return 2

    cited_keys: Optional[Set[str]] = None
    if aux_path is not None:
        if not aux_path.exists():
            _print_error(f"{aux_path} not found")
            return 2
        if aux_path.suffix != ".aux":
            _print_error(f"{aux_path} is not an .aux file")
            return 2
        try:
            cited_keys = _cited_keys_from_aux(aux_path)
        except FileNotFoundError as exc:
            _print_error(f"{exc} not found")
            return 2

    try:
        manager = _load_manager(bib_path, profile)
    except StyleConfigError as exc:
        _print_error(exc.message)
        return 2

    original = bib_path.read_text(encoding="utf-8")
    source_name = str(bib_path)
    errors = 0
    warnings = 0
    parse_failures = []
    failed_keys = set()
    for block in manager.failed_blocks:
        key = _failed_block_key(block)
        if cited_keys is not None and key not in cited_keys:
            continue
        failed_keys.add(key)
        parse_failures.append(
            {"key": key, "reason": block.__class__.__name__}
        )
        _print_error(
            f"{source_name}: {key} failed to parse ({block.__class__.__name__})"
        )
        errors += 1

    entries = manager.bibliography_entity
    if cited_keys is not None:
        entries = [entry for entry in entries if entry.key in cited_keys]
        present = {entry.key for entry in entries} | failed_keys
        for key in sorted(cited_keys - present):
            _print_error(f"{source_name}: {key} not found in bibliography")
            errors += 1

    entry_reports, entry_errors, entry_warnings = _collect_entry_issues(
        entries, source_name, manager.rules.max_authors
    )
    errors += entry_errors
    warnings += entry_warnings

    from ures.literature.citation.middlewares import (
        OutputCleanupNoneResultMiddleware,
        OutputLimitMaxAuthors,
    )

    library = _subset_library(manager.bibliograph_library, cited_keys)
    middlewares = [
        OutputLimitMaxAuthors(rule_register=manager.rules),
        OutputCleanupNoneResultMiddleware(rule_register=manager.rules),
        bibtexparser.middlewares.MergeNameParts(),
        bibtexparser.middlewares.MergeCoAuthors(),
        bibtexparser.middlewares.SortFieldsAlphabeticallyMiddleware(),
        bibtexparser.middlewares.SortBlocksByTypeAndKeyMiddleware(),
    ]
    normalized = bibtexparser.write_string(library, append_middleware=middlewares)
    drift = cited_keys is None and normalized != original
    if not write_back and drift:
        _print_error(f"{source_name}: formatting differs from normalized output")
        errors += 1
    if write_back and dest_path is not None:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(normalized, encoding="utf-8")

    report_path = str(dest_path) if write_back and dest_path is not None else source_name
    _print_error_summary = (
        f"{source_name}: {errors} error{'s' if errors != 1 else ''}, "
        f"{warnings} warning{'s' if warnings != 1 else ''}"
    )
    print(_print_error_summary, file=sys.stderr)
    _write_report(
        {
            "profile": profile,
            "style": manager.rules.style_name,
            "files": [
                {
                    "path": report_path,
                    "drift": drift,
                    "parse_failures": parse_failures,
                    "entries": entry_reports,
                }
            ],
        }
    )
    if errors:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ures-bib")
    sub = parser.add_subparsers(dest="command")

    init_cmd = sub.add_parser("init")
    init_cmd.add_argument("--style", choices=PACKAGED_STYLES, default="acm")

    format_cmd = sub.add_parser("format")
    format_cmd.add_argument("bib_file")
    format_cmd.add_argument("--profile", default="library")
    format_cmd.add_argument("--aux")
    format_cmd.add_argument("--output")

    check_cmd = sub.add_parser("check")
    check_cmd.add_argument("bib_file")
    check_cmd.add_argument("--profile", default="library")
    check_cmd.add_argument("--aux")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(sys.stderr)
        _print_error("missing command")
        return 2
    try:
        if args.command == "init":
            return cmd_init(args.style)
        bib_path = Path(args.bib_file)
        aux_path = Path(args.aux) if args.aux else None
        output_path = Path(args.output) if getattr(args, "output", None) else None
        return cmd_process(
            bib_path,
            args.profile,
            write_back=args.command == "format",
            aux_path=aux_path,
            output_path=output_path,
        )
    except StyleConfigError as exc:
        _print_error(exc.message)
        return 2
    except Exception as exc:
        _print_error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
