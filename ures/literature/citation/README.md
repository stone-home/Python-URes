This README details the design and usage of the `ures.literature.citation` module, a comprehensive system for parsing, validating, and managing BibTeX bibliographies based on document usage.

This system is designed to load one or more master `.bib` files, validate them against a strict, style-specific set of rules, and then cross-reference them against citations found in LaTeX source files (`.tex`, `.bbl`). The primary goal is to produce a clean, validated, and properly formatted final `.bib` file that contains *only* the entries that were actually cited in the document.

## Design and Architecture

The module operates on several core components that work together:

1.  **Managers (The Facade and the Engine):**

      * **`CitationManager`:** This is the main user-facing class. It acts as a facade, coordinating the other components. It initializes the `BibManager`, runs the extractors, and links cited keys back to the full bibliography entries.
      * **`BibManager`:** This is the core bibliography engine. It is responsible for loading, parsing, and validating the master `.bib` files. It utilizes a middleware pipeline to process every entry it loads.

2.  **Citation Extractors (`extractors.py`):**

      * These components are responsible for finding *which* citation keys are used in the source documents.
      * **`TexCitationExtractor`:** Parses `.tex` files using regex to find all citation commands (e.g., `\cite`, `\citep`, `\nocite`) and extracts the keys.
      * **`BBLCitationExtractor`:** Parses `.bbl` files to find `\bibitem` entries, providing a reliable source of all citations included in the final compiled document.
      * **`CitationInfo`:** A data structure used to store the extracted key and a list of its locations (file and line number) found by the extractors.

3.  **Rule System (`rules/`):**

      * This system defines the validation and formatting logic.
      * **`BibRuleRegister`:** A class that loads and manages citation styles (e.g., "default" or "acm").
      * **`BibTypeRule`:** A data class defining the rules for a specific entry type (like `article` or `inproceedings`). This includes required fields, optional fields, forbidden fields, and field mappings (e.g., mapping `journaltitle` to `journal`).
      * **Rule Sets:** The module provides predefined rule sets, such as `DefaultRules` and `ACMBibStyle`, which specify different requirements. For example, the `acm` style sets the `proceedings_style` formatting rule to "proceedings".

4.  **Middleware Pipeline (`middlewares.py`):**

      * This is the core normalization and validation engine used by `BibManager`. When a `.bib` file is loaded, every entry is passed through this pipeline.
      * **Normalization:** A series of middlewares clean the data:
          * `TypeNormalizationMiddleware`: Standardizes entry types (e.g., maps `conference` to `inproceedings`).
          * `FieldNormalizationMiddleware`: Maps non-standard field keys to standard ones (e.g., `location` to `address`) and normalizes page dashes.
          * `DateSpiltToYearMonthDayMiddleware`: Parses `date` fields into separate `year`, `month`, and `day` fields.
          * `ProceedingsNormalizationMiddleware`: Automatically reformats `booktitle` fields based on the loaded rule (e.g., stripping "In Proceedings of the" and reapplying the correct prefix like "Proceedings of the" or just "In").
          * `PublisherNormalizationMiddleware`: Standardizes publisher names (e.g., ensuring "IEEE" becomes "IEEE Inc.").
      * **Validation:**
          * `RuleBasedValidationMiddleware`: After normalization, this middleware checks the entry against the loaded `BibTypeRule`. It verifies that all `required_fields` are present and adds metadata (`is_valid` and `missing_fields`) to the entry.
      * **Output Processing:** When *saving* the final library, a separate set of output middlewares is used:
          * `OutputLimitMaxAuthors`: Enforces a maximum number of authors (e.g., 5), appending "and others" if the limit is exceeded.
          * `OutputOnlyDesiredFieldsMiddleware`: Removes any fields not listed as required or optional by the rules, ensuring a clean output.
          * `OutputCleanupNoneResultMiddleware`: Removes any fields that have empty or `None` values.

## Usage Workflow

The intended workflow involves loading a master bibliography, identifying all used citations from LaTeX source files, and exporting a clean, validated subset of the library. This process is demonstrated in the `test.ipynb` notebook.

**Step 1: Initialize the Citation Manager**

Create an instance of `CitationManager`. Pass the path to your master `.bib` file (or a list of files) and specify the desired validation style (e.g., "acm" or "default").

```python
from ures.literature.citation import CitationManager
from pathlib import Path

project_dir = Path('./my_project')
bib_file = project_dir / 'references.bib'
tex_file = project_dir / 'main.tex'
bbl_file = project_dir / 'main.bbl'

# Load the master bibliography and apply the 'acm' style rules
cite_man = CitationManager(bib_file, bibliography_style='acm')
```

**Step 2: Import Citations from Source Files**

Use the `.import_citations()` method, passing it a list of your LaTeX source files (`.tex` and/or `.bbl`). This will read the files, extract all unique citation keys, and link them to the validated entries loaded from your master `.bib` file.

```python
# Extract all citation keys from the .tex and .bbl files
imported_citations = cite_man.import_citations(files=[tex_file, bbl_file])
```

**Step 3: (Optional) Review Failures**

You can inspect any entries from your master `.bib` file that failed to parse (due to syntax errors) or failed validation (due to missing required fields).

```python
# Check for BibTeX syntax errors
failed_blocks = cite_man.bib_library.failed_blocks
if failed_blocks:
    print("Entries that failed to parse:")
    cite_man._bib_manager.display_failed_entities()

# Check for entries that failed rule-based validation
print("\nEntries that are invalid (missing fields):")
cite_man.display_invalid_citations()
```

**Step 4: Save the Clean Bibliography**

Finally, call `.save_bibliography()` to create a new, clean `.bib` file. This output file will contain *only* the entries that were successfully parsed, validated, and found in your source files. The entries will also be processed by the output pipeline (limiting authors, removing extra fields, etc.).

```python
# Save the final, clean, and cited-only bibliography
cite_man.save_bibliography(file_path="output/references_clean.bib")
```
