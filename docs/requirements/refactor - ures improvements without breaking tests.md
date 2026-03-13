---
title: refactor - ures improvements without breaking tests
id: req7k2m9p4x
create: 2025-03-13
tags:
  - type/code-requirement
aliases: []
sources: []
type: permanent
url:
cssclasses: []
project: ""
status: done
priority: medium
release: v0.0.0
---
# Commit Message
`refactor: ures package improvements (typos, types, logging); tests unchanged`

# PR Description
**Title:** `refactor: ures package improvements without breaking tests`
**Summary:** Improves the ures package with a typo fix, modern type hints, logging instead of print, and minor code cleanups. No test files were modified; all existing test cases continue to pass.

---
# Refactor: ures Package Improvements Without Breaking Tests

## 1. Requirements & Context
- Fix a known typo in the literature citation rules API without changing behavior.
- Align type hints with Python 3.9+ style (built-in generics, union syntax) where applicable.
- Replace print with logging in files module for consistency and production use.
- Use module logger name (`__name__`) instead of `__file__` in decorator for correct log hierarchy.
- Move inline imports to module top in timedate for clarity.
- Minor cleanups (e.g. `in self.children` instead of `in self.children.keys()`) for idiomatic Python.
- All changes must preserve existing test behavior; no edits under `tests/`.

## 2. Execution Plan
- [x] Fix typo: `get_defulat_bib_type_mapping` → `get_default_bib_type_mapping` in `ures/literature/citation/rules/__init__.py` and call site in `ures/literature/citation/middlewares.py`.
- [x] Add/update type hints: `string2date(date_string: str)`, `format_memory(nbytes: int | None)` in `ures/string.py`; `list[str]`, `list[str] | None` in `ures/files.py`; `field: str | None = None` in `ures/tools/enum.py`; modernize `ures/data_structure/tree.py` (e.g. `dict[str, TreeNode]`, `TreeNode | None`, `list[list[Any]]`).
- [x] In `ures/data_structure/tree.py`, use `child.id not in self.children` and `child.id in self.children` instead of `self.children.keys()`.
- [x] In `ures/files.py`, add logging and use `logger.warning(...)` in `list_directories` when path does not exist; remove print.
- [x] In `ures/tools/decorator.py`, use `logging.getLogger(__name__)` instead of `logging.getLogger(__file__)`.
- [x] In `ures/timedate.py`, move `import sys` to top of file and remove inline import from `time_now()`.
- [x] Run full pytest suite and confirm all test cases pass (no changes to `tests/`).
