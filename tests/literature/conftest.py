import json
import os
import shutil
from pathlib import Path

import pytest

from ures.literature.citation.bib_cli import REPORT_FILENAME, main

REFERENCES_BIB = Path(__file__).resolve().parent / "references.bib"
SKIP_ENTRY_TYPES = {"comment", "preamble", "string"}


def source_entry_inventory(path: Path) -> list[tuple[str, str]]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("@") or "{" not in stripped:
            continue
        type_part, key_part = stripped.split("{", 1)
        entry_type = type_part[1:].strip().lower()
        if entry_type in SKIP_ENTRY_TYPES:
            continue
        entries.append((entry_type, key_part.rstrip(",").strip()))
    return entries


def load_report(directory: Path) -> dict:
    report_path = directory / REPORT_FILENAME
    assert report_path.is_file()
    return json.loads(report_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def references_bib_path():
    assert REFERENCES_BIB.is_file()
    return REFERENCES_BIB


@pytest.fixture(scope="session")
def references_inventory(references_bib_path):
    inventory = source_entry_inventory(references_bib_path)
    assert inventory
    return inventory


@pytest.fixture
def corpus_bib(tmp_path, references_bib_path):
    dest = tmp_path / "references.bib"
    shutil.copy2(references_bib_path, dest)
    return dest


@pytest.fixture(scope="module")
def acm_library_check(tmp_path_factory, references_bib_path):
    work = tmp_path_factory.mktemp("acm-library-check")
    dest = work / "references.bib"
    shutil.copy2(references_bib_path, dest)
    cwd = Path.cwd()
    try:
        os.chdir(work)
        code = main(["check", str(dest), "--profile", "library"])
    finally:
        os.chdir(cwd)
    return {
        "code": code,
        "report": load_report(work),
        "bib_path": dest,
        "work": work,
        "original_bytes": references_bib_path.read_bytes(),
    }
