# Bib standardizer for `BibManager`

Date: 2026-09-17
Amended: 2026-09-21 (`--aux`, `format --output`)
Status: approved for spec review
Scope: overlay JSON style data, profiles, a small CLI, and a lint report on top of the existing citation pipeline. Do not replace `BibManager`, middlewares, or `BibRuleRegister`.

## Problem

`ures.literature.citation` already normalizes BibTeX (field names, types, pages, dates, proceedings prefixes, publishers) and can export a cleaned library. It is not a CI tool:

- Rules live only in Python (`DefaultRules`, `ACMBibStyle`).
- Default required fields are too strict for ML+Sys libraries (`month`, `issn`, `address`).
- Export drops fields that are not required/optional.
- There is no `format` / `check` CLI, no overlay config, and no human-readable failure output.

The tool should behave like a bib formatter/linter: write a normalized copy, CI check, one JSON file to customize ACM or IEEE. `format` never overwrites the input `.bib`.

## Goals

1. Configure required/suggested fields via JSON.
2. Ship ACM and IEEE baselines. Local `.bibstyle.json` is either a full dump from `init` or an `extends` overlay with add/remove.
3. Keep field-level standardization (proceedings is one of those fields).
4. Author-count limit is configurable in JSON.
5. `format` writes a normalized copy (default `{stem} - formatted.bib` beside the input); `check` writes neither the input nor that copy. Both print clear errors and set exit codes for CI. Optional `--aux` limits both commands to keys recorded in the LaTeX `.aux` file.
6. Extra fields are always kept. Missing fields go to stderr and a report file.
7. Strictness is a profile. Only camera-ready promotes all suggested fields to errors.
8. Change existing Python by loading data and adding a thin CLI, not by rewriting the pipeline.

## Non-goals

- New GitHub Actions workflow in this repository.
- Recursing directories, multiple input `.bib` files per invocation, or extra CLI flags (`--config`, `--report`, `--max-authors`, `show-config`). `--aux` and `format --output` are in scope.
- Dropping unknown fields.
- A third venue style (USENIX uses ACM's bst; treat it as ACM).
- Replacing bibtexparser, middlewares, or `CitationManager` (cited-only export stays as-is).

## Architecture

Keep the current engine. Add a style-data layer and a CLI that calls it.

```text
FILE.bib + packaged style + optional cwd bibstyle.json + --profile
    + optional --aux (cite keys from \citation / nested \@input)
    → BibRuleRegister (existing rules filled/overlaid from JSON)
    → BibManager.load_from_file (existing normalize middlewares)
    → if --aux: keep only cited keys; missing cited keys are errors
    → validate required / suggested for the (possibly filtered) entries
    → format: write --output (default `{stem} - formatted.bib` beside FILE.bib)
    → check without --aux: compare in-memory normalized bib to FILE.bib; no write
    → check with --aux: no whole-file drift; validate cited keys only; no write
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
- `doi` or `url` for ACM `submission` / identifier suggested. IEEE `submission` and camera-ready only require `url` (IEEEtran has no `doi` field). ACM camera-ready still accepts `url` without `doi`.

`month` is never required. Current Python `BasicRequiredFields` that include `month` must not remain the library default once JSON styles are loaded.

### Sources (verified 2026-09-17)

acm.org HTML is behind Cloudflare from this environment and must not be used as a fetch target. Camera-ready fields come from the style files that LaTeX actually runs, pulled with a 20s `curl` timeout:

- ACM: `ACM-Reference-Format.bst` v2.2, acmart 2.19, dated 2026-08-12, from `https://raw.githubusercontent.com/borisveytsman/acmart/primary/ACM-Reference-Format.bst`. Required means `output.check` / `bibinfo.output.check`. Year empty is a bst warning (`output.year.check`) and is still required in our JSON. `doi`/`url`/`eprint` print if present; they are not `output.check`.
- IEEE: `IEEEtran.bst` 1.14 (2015/08/26) from `https://mirrors.mit.edu/CTAN/macros/latex/contrib/IEEEtran/bibtex/IEEEtran.bst`. Required means `output.warn`. The ENTRY list has `url` and **no `doi` key**; DOI in a `.bib` is kept as an extra field but is never an IEEE camera-ready error. Conference `booktitle` convention (`Proc. {IEEE} ...`) is from the IEEEtran HOWTO examples; the bst itself only prepends “in” and emphasizes `booktitle`.

### ACM vs IEEE packaged data

Shared **library** required set: `author`, `title`, `year`, plus the venue field. Camera-ready promotes the suggested column (and ACM bst-required publisher/address).

| Type | Library required extra | ACM suggested (camera-ready → error) | IEEE suggested (camera-ready → error) |
|------|------------------------|--------------------------------------|---------------------------------------|
| `article` | `journal` | `volume`, `number`, `pages` **or** `articleno`, `doi` **or** `url` | `volume`, `pages`, `url` |
| `inproceedings` | `booktitle` | `publisher`, `address`, `pages` **or** `articleno`, `doi` **or** `url` | `pages`, `address`, `url` |
| `preprint` | (none beyond the three) | `eprint`+`archivePrefix` **or** `doi`/`url` (ACM bst aliases `@preprint` to `manual` but still recognizes `eprint`/`archiveprefix`) | `url` (keep `eprint`/`doi` if present; not IEEE-required) |
| `book` | ACM: `publisher` **and** `address`. IEEE: `publisher` | `isbn` | `address`, `url` |
| `techreport` | `institution` | `address`, `doi` **or** `url` | `address`, `url` |
| `thesis` | ACM: `school`. IEEE: `school` | `address`, `doi` **or** `url` | `address`, `url` |
| other existing types | `author`/`title`/`year`; `online` also `url` | current optional fields become suggested | same |

ACM `@conference` is an alias of `@inproceedings`. Map it as we already do.

`proceedings_style` normalizes the **value of `booktitle`**, not the bst wrapper:

- ACM `"proceedings"`: store `Proceedings of the …` (ACM bst then prints `In` + emphasized booktitle).
- IEEE `"proc"`: store `Proc. …` (IEEEtran HOWTO convention; bst does not insert `Proc.`).

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
ures-bib format FILE.bib [--aux FILE.aux] [--output PATH] [--profile library|submission|camera-ready]
ures-bib check  FILE.bib [--aux FILE.aux] [--profile library|submission|camera-ready]
```

- `init`: write packaged baseline JSON to `./bibstyle.json`. Default `--style acm`. If the file exists, print an error and exit 2 unless the user deletes it first. No `--force` flag (keep the CLI small; overwrite is a manual delete).
- `format`: never overwrite `FILE.bib`. Normalize and write `--output`. Default `--output` is `{stem} - formatted{suffix}` in the same directory as `FILE.bib` (so `references.bib` → `references - formatted.bib`). If that destination already exists, overwrite it. If `--output` resolves to the same path as `FILE.bib`, exit 2. Always write `./bib-lint-report.json`. Print issues. Exit 1 if any error remains after write.
- `check`: do not write `FILE.bib` or a formatted copy. `--output` is not a `check` flag. Without `--aux`, if the serialized result differs from `FILE.bib`, that is an error (`formatting differs from normalized output`). With `--aux`, skip whole-file drift. Write the same report path. Exit 1 on any error including (no-aux) drift.
- `--aux`: optional. If omitted, process every entry in `FILE.bib`. If present, read BibTeX `.aux` `\citation{...}` keys (comma-separated) and follow `\@input{...}` nested aux files relative to the aux that named them. Only those keys are validated (`check`) or written (`format`). A cited key missing from the bibliography is an error. `\citation{*}` means the whole library (same as omitting `--aux` for filtering). Missing `--aux` path, non-`.aux` suffix, or missing nested `\@input` target: exit 2. Do not parse `.bcf`. `\bibdata` is not used to locate `.bib` files.

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
error: paper.aux not found
error: refusing to overwrite input refs.bib
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
- BibTeX failed blocks: each is an error in stderr and `parse_failures`, except with `--aux` only blocks whose key is a cited key. Other entries still process. Failed blocks are not rewritten into fake entries.
- `check` drift without `--aux`: file-level error, not per-key. With `--aux`, do not compare the whole `.bib` to a cited subset.
- `format` always attempts write of entries that parsed (the cited subset when `--aux` is set) to `--output`; failed blocks remain a problem the user must fix in the source (do not silently drop them without reporting).
- Cited key absent from `FILE.bib`: `error: FILE.bib: {key} not found in bibliography`.
- Warnings never change the exit code by themselves.

## Testing

Do not weaken or rewrite existing middleware/rule tests. Add:

- Style load, overlay add/remove, rejected `required_remove` of the three core fields.
- Profile promotion (`library` vs `submission` vs `camera-ready`).
- `doi`/`url` and `pages`/`articleno` equivalence.
- Export keeps extra fields; `max_authors=0` does not truncate.
- CLI: `init` writes JSON that loads again as that style; existing `bibstyle.json` makes `init` exit 2; `check` on an unnormalized file exits 1 and prints `error:`; `format` writes `{stem} - formatted.bib` and leaves the input bytes unchanged; `check` of that formatted copy on a complete entry can exit 0; `format --aux` writes only cited keys; `check --aux` does not fail on unused incomplete entries; unknown profile prints `error:` and exits 2.

Use temporary directories. No coverage gate. This spec does not authorize editing tests in the same developer conversation as production code.

## Compatibility

- `CitationManager` and public middleware class names stay.
- `BibManager(..., bibliography_style="acm"|"ieee"|"default")` remains. `"default"` and `"acm"` both load packaged ACM JSON (library required/suggested), not the old Python list that required `month`. `"ieee"` loads packaged IEEE JSON.
- Existing `CitationManager(..., bibliography_style=...)` keeps the same argument; it picks up the new data through `BibManager`.

## Implementation constraints

- Diff stays in citation rules, export middleware selection, packaged JSON, the CLI module, and an aux citation extractor.
- No second bib parser, no new validation framework, no rewrite of proceedings regexes beyond adding the `proc` prefix key. Aux parsing is `\citation` / `\@input` only.
- No dependency beyond the existing Poetry set (`bibtexparser`, stdlib JSON).
