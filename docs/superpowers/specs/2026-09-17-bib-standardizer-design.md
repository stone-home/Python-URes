# Bib standardizer for `BibManager`

Date: 2026-09-17  
Status: approved for spec review  
Scope: overlay JSON style data, profiles, a small CLI, and a lint report on top of the existing citation pipeline. Do not replace `BibManager`, middlewares, or `BibRuleRegister`.

## Problem

`ures.literature.citation` already normalizes BibTeX (field names, types, pages, dates, proceedings prefixes, publishers) and can export a cleaned library. It is not a CI tool:

- Rules live only in Python (`DefaultRules`, `ACMBibStyle`).
- Default required fields are too strict for ML+Sys libraries (`month`, `issn`, `address`).
- Export drops fields that are not required/optional.
- There is no `format` / `check` CLI, no overlay config, and no human-readable failure output.

The tool should behave like a bib formatter/linter: local rewrite, CI check, one JSON file to customize ACM or IEEE.

## Goals

1. Configure required/suggested fields via JSON.
2. Ship ACM and IEEE baselines. Local `bibstyle.json` is either a full dump from `init` or an `extends` overlay with add/remove.
3. Keep field-level standardization (proceedings is one of those fields).
4. Author-count limit is configurable in JSON.
5. `format` writes a normalized `.bib`; `check` does not. Both print clear errors and set exit codes for CI.
6. Extra fields are always kept. Missing fields go to stderr and a report file.
7. Strictness is a profile. Only camera-ready promotes all suggested fields to errors.
8. Change existing Python by loading data and adding a thin CLI, not by rewriting the pipeline.

## Non-goals

- New GitHub Actions workflow in this repository.
- Recursing directories, multiple input files per invocation, or extra CLI flags (`--config`, `--report`, `--max-authors`, `show-config`).
- Dropping unknown fields.
- A third venue style (USENIX uses ACM's bst; treat it as ACM).
- Replacing bibtexparser, middlewares, or `CitationManager` (cited-only export stays as-is).

## Architecture

Keep the current engine. Add a style-data layer and a CLI that calls it.

```text
FILE.bib + packaged style + optional cwd bibstyle.json + --profile
    → BibRuleRegister (existing rules filled/overlaid from JSON)
    → BibManager.load_from_file (existing normalize middlewares)
    → validate required / suggested for the profile
    → format: write FILE.bib + report + stderr
    → check:  compare in-memory normalized bib to FILE.bib; no write of FILE.bib
```

Resolution order for style data:

1. If `./bibstyle.json` is absent, use packaged `acm.json`.
2. If it is a **full style** (`name` plus `entry_types.*.required` / `suggested`, no `extends`): use the file as the complete baseline. This is what `init` writes.
3. If it is an **overlay** (`extends` plus `*_add` / `*_remove`): start from packaged `acm` or `ieee`, then apply add/remove.
4. A file that mixes `extends` with complete `required`/`suggested` lists is a config error (exit 2).
5. Apply `--profile` to decide which suggested fields become errors.

Do not walk parent directories. Do not take `--config`.

Code touch list (small):

- `BibRuleRegister`: load packaged JSON, apply overlay, expose required vs suggested for a profile.
- `BibTypeRule` / `OutputRules` / `FormattingRules`: use existing fields; suggested lists may live on the rule or beside it. Do not invent a second rule engine.
- `export_to_file`: do not run `OutputOnlyDesiredFieldsMiddleware` by default, so extra fields survive.
- `OutputLimitMaxAuthors`: apply only when `max_authors > 0`.
- `ProceedingsNormalizationMiddleware` prefix map: add `"proc"` → `"Proc. "` for IEEE. Do not rewrite matching logic.
- New packaged JSON under `ures/literature/citation/styles/`.
- New thin CLI module and Poetry console script `ures-bib`.
- `RuleBasedValidationMiddleware` (or a small helper used by CLI): record missing required and missing suggested instead of a single `is_valid` bit if that bit is not enough for the report. Prefer extending the existing validator over adding a parallel one.

Do not rewrite `FieldNormalizationMiddleware`, page dash logic, date split, publisher aliases, or type mapping tables except where JSON supplies the same mappings the register already has.

## Style JSON

Packaged files: `acm.json`, `ieee.json`. Same schema.

```json
{
  "name": "acm",
  "max_authors": 0,
  "proceedings_style": "proceedings",
  "entry_types": {
    "article": {
      "required": ["author", "title", "year", "journal"],
      "suggested": ["volume", "number", "pages", "doi"]
    },
    "inproceedings": {
      "required": ["author", "title", "year", "booktitle"],
      "suggested": ["pages", "doi", "publisher"]
    }
  }
}
```

`max_authors: 0` means no author cap (no warning, no truncate). A positive cap warns when exceeded; `format` also truncates to that cap.

Existing type mappings (`conference` → `inproceedings`, …) and field mappings (`journaltitle` → `journal`, …) stay in `BibRuleRegister` unless a later overlay needs them. v1 overlay does not customize mappings.

### Profiles

`--profile` is CLI-only. It is not stored in local JSON.

| Profile | Missing `required` | Missing `suggested` |
|---------|--------------------|---------------------|
| `library` (default) | error | warning |
| `submission` | error | `doi` or `url` becomes error; other suggested stay warning |
| `camera-ready` | error | all suggested become error |

Equivalence (either field satisfies the requirement):

- `pages` or `articleno`
- `doi` or `url` (`submission` and identifier suggested). Camera-ready ACM still accepts `url` if `doi` is absent; do not fail an entry that has a resolvable `url` but no `doi`.

`month` is never required. Current Python `BasicRequiredFields` that include `month` must not remain the library default once JSON styles are loaded.

### ACM vs IEEE packaged data

Shared required set for the types below: `author`, `title`, `year`, plus the venue field.

| Type | Required extra | ACM suggested | IEEE suggested |
|------|----------------|---------------|----------------|
| `article` | `journal` | `volume`, `number`, `pages`, `doi` | same |
| `inproceedings` | `booktitle` | `pages`, `doi`, `publisher`, `address` | `pages`, `doi`, `address` |
| `preprint` | (none beyond the three) | `eprint`+`archivePrefix`, or `doi`/`url` | same |
| `book` | `publisher` | `isbn`, `address` | same |
| `techreport` | `institution` | `address`, `url` | same |
| `thesis` | `school` **or** `publisher` (either satisfies) | `address`, `url` | same |
| other existing types (`misc`, `online`, `software`, …) | keep current type in the register; required = `author`/`title`/`year` plus the current distinguishing field if it is a true locator (`url` for `online`) | current optional fields become suggested | same |

`proceedings_style`: ACM `"proceedings"` (already `"Proceedings of the "`); IEEE `"proc"` (new prefix `"Proc. "`).

Default when no `bibstyle.json`: ACM.

### Local overlay

`init` writes a **full style** (the packaged file). To keep the packaged baseline and only tweak it, use this overlay shape instead.

`bibstyle.json` in the current working directory:

```json
{
  "extends": "ieee",
  "max_authors": 8,
  "proceedings_style": "proc",
  "entry_types": {
    "inproceedings": {
      "required_add": ["pages"],
      "suggested_add": ["doi"],
      "suggested_remove": ["address"]
    }
  }
}
```

Allowed keys: `extends`, `max_authors`, `proceedings_style`, `entry_types.<type>.required_add|required_remove|suggested_add|suggested_remove`.

Unknown `entry_type` in overlay creates that type with required `author`/`title`/`year` then applies add/remove.

`required_remove` of `author`, `title`, or `year` is a configuration error (exit 2).

Lists are merged in order: baseline, then remove, then add. Duplicates collapse. `extends` default is `acm` if omitted.

## CLI

Poetry script: `ures-bib`.

```text
ures-bib init [--style acm|ieee]
ures-bib format FILE.bib [--profile library|submission|camera-ready]
ures-bib check  FILE.bib [--profile library|submission|camera-ready]
```

- `init`: write packaged baseline JSON to `./bibstyle.json`. Default `--style acm`. If the file exists, print an error and exit 2 unless the user deletes it first. No `--force` flag (keep the CLI small; overwrite is a manual delete).
- `format`: normalize and write `FILE.bib`. Always write `./bib-lint-report.json`. Print issues. Exit 1 if any error remains after write.
- `check`: do not write `FILE.bib`. Normalize in memory. If the serialized result differs from the file, that is an error (`formatting differs from normalized output`). Write the same report path. Exit 1 on any error including drift.

One positional `.bib` file for `format`/`check`. Missing or non-`.bib` path is exit 2.

`--profile` default: `library`.

## Stderr and exit codes

Every issue is one line on stderr, prefixed `error:` or `warning:`. Last line is a summary on stderr:

```text
error: refs.bib: Smith2024 missing required field booktitle
warning: refs.bib: Jones2023 missing doi
error: refs.bib: formatting differs from normalized output
refs.bib: 2 errors, 1 warning
```

Config/usage failures also use `error:` and exit 2:

```text
error: unknown style 'foo' (use acm or ieee)
error: bibstyle.json: cannot remove required field author
error: refs.bib not found
error: failed to parse bibstyle.json: ...
```

| Code | When |
|------|------|
| 0 | No errors. Warnings allowed. |
| 1 | Entry errors (missing required, parse failure of an entry, `check` drift) and/or `format` completed with remaining required-field errors. |
| 2 | Usage, missing file, bad JSON, illegal overlay, unknown style/profile. |

Do not dump a Python traceback unless the failure is an unexpected exception; then still print `error:` first.

`bib-lint-report.json` is the machine-readable copy (profile, style, path, drift, parse_failures, per-entry missing_required / missing_suggested / author_count / severity). Humans should not need it to understand a CI failure.

## Error handling

- Invalid JSON, unknown `extends`/`--style`/`--profile`: exit 2, one `error:` line.
- BibTeX failed blocks: each is an error in stderr and `parse_failures`. Other entries still process. Failed blocks are not rewritten into fake entries.
- `check` drift: file-level error, not per-key.
- `format` always attempts write of entries that parsed; failed blocks remain a problem the user must fix in the source (do not silently drop them without reporting).
- Warnings never change the exit code by themselves.

## Testing

Do not weaken or rewrite existing middleware/rule tests. Add:

- Style load, overlay add/remove, rejected `required_remove` of the three core fields.
- Profile promotion (`library` vs `submission` vs `camera-ready`).
- `doi`/`url` and `pages`/`articleno` equivalence.
- Export keeps extra fields; `max_authors=0` does not truncate.
- CLI: `init` writes JSON that loads again as that style; existing `bibstyle.json` makes `init` exit 2; `check` on an unnormalized file exits 1 and prints `error:`; `format` then `check` on a clean file exits 0; unknown profile prints `error:` and exits 2.

Use temporary directories. No coverage gate. This spec does not authorize editing tests in the same developer conversation as production code.

## Compatibility

- `CitationManager` and public middleware class names stay.
- `BibManager(..., bibliography_style="acm"|"ieee"|"default")` remains. `"default"` and `"acm"` both load packaged ACM JSON (library required/suggested), not the old Python list that required `month`. `"ieee"` loads packaged IEEE JSON.
- Existing `CitationManager(..., bibliography_style=...)` keeps the same argument; it picks up the new data through `BibManager`.

## Implementation constraints

- Diff stays in citation rules, export middleware selection, packaged JSON, and a new CLI module.
- No second parser, no new validation framework, no rewrite of proceedings regexes beyond adding the `proc` prefix key.
- No dependency beyond the existing Poetry set (`bibtexparser`, stdlib JSON).
