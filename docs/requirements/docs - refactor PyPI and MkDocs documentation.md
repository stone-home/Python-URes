---
title: docs - refactor PyPI and MkDocs documentation
id: req8pypi3d
create: 2026-03-13
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
`docs: refactor PyPI long description and MkDocs nav for clarity and correctness`

# PR Description
**Title:** `docs: refactor PyPI and MkDocs documentation`
**Summary:** Reworks the PyPI-facing documentation (docs/index.md), fixes the broken LICENSE link, aligns overview and features with actual modules, expands MkDocs navigation to expose module and API docs, and updates pyproject.toml description. Documentation is deployed via .github/workflows/gh-page.yaml (reusable Action-Stone-DevOps workflow).

---
# Docs: Refactor PyPI and MkDocs Documentation

## 1. Requirements & Context
- PyPI long description is sourced from `docs/index.md` via `pyproject.toml` readme; the same file serves as the MkDocs home page deployed to GitHub Pages by `.github/workflows/gh-page.yaml`.
- Fix incorrect or broken content: LICENSE link pointed to a Google search; "Advanced Plotting" was listed but no such module exists.
- Align overview and key features with the real module set: Docker, Markdown/Zettelkasten, data structures, core utilities (files, timedate, string, secrets, network), and optional literature tooling.
- Improve discoverability: MkDocs nav previously had only Home and API; module docs and API sub-pages were not linked.
- Keep PyPI page self-contained (no reliance on relative doc links that break on PyPI).
- Align `pyproject.toml` short description with the refactored overview.

## 2. Execution Plan
- [x] Refactor `docs/index.md`: overview, key features, module summaries (Docker, Markdown & Zettelkasten, data structures, core utilities, literature tooling), fix LICENSE to GitHub repo URL, keep Development and License sections.
- [x] Expand `mkdocs.yml` nav: add Modules section (core-utilities, data-structure, docker, memory, markdown, literature-search, literature-citation, tools) and API section with overview plus per-module API pages (files, string, timedate, network, markdown, docker, data_structure/tree, data_structure/bi_directional_links).
- [x] Update `pyproject.toml` description to match the new overview.
- [x] Run `poetry run mkdocs build` to verify the site builds with the new nav.
