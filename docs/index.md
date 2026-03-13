# Utility for Research (URes)

[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/stone-home/Python-URes)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/stone-home/Python-URes/blob/main/LICENSE)
[![Super-Linter](https://github.com/stone-home/Python-URes/actions/workflows/linter.yaml/badge.svg)](https://github.com/stone-home/Python-URes/actions/workflows/linter.yaml)
[![Trufflehog](https://github.com/stone-home/Python-URes/actions/workflows/secret-check.yaml/badge.svg)](https://github.com/stone-home/Python-URes/actions/workflows/secret-check.yaml)
[![Code Testing](https://github.com/stone-home/Python-URes/actions/workflows/test.yaml/badge.svg)](https://github.com/stone-home/Python-URes/actions/workflows/test.yaml)

---

## Overview

URes (Utility for Research) is a Python library that centralizes **reusable utilities for research and development**. It provides batteries-included helpers for Docker workflows, Markdown and Zettelkasten notes, data structures, filesystem and network tasks, time/date conversion, string manipulation, and more.

## Key Features

- **Docker management**: Build, run, and orchestrate Docker images and containers with structured configuration objects.
- **Markdown & Zettelkasten**: Work with Markdown files and front matter, and manage Zettelkasten-style notes with required metadata.
- **Data structures**: Use tree and bi-directional link structures for organizing and traversing complex data.
- **Core utilities**: File, time/date, string, secrets, and network helpers used across projects.
- **Literature tooling**: Search across literature sources and manage citation data (when those modules are enabled).



## Installation

```bash
pip install ures
```

## Modules

### Docker

The `ures.docker` module provides a high-level interface for managing Docker images and containers.

**Key Features**:

  - Programmatically build Docker images from a `BuildConfig` object.
  - Run containers with detailed runtime configurations using `RuntimeConfig`.
  - Orchestrate the building of multiple images with dependencies using `ImageOrchestrator`.

**Example**:

```python
from ures.docker import Image, BuildConfig, Containers

# Define build configuration
build_config = BuildConfig(
    base_image="python:3.10-slim",
    python_dependencies=["flask", "gunicorn"],
    copies=[{"src": "./app", "dest": "/app"}],
    entrypoint=["gunicorn", "-b", "0.0.0.0:80", "app:app"]
)

# Build the image
image = Image("my-web-app")
image.build_image(build_config, dest=".")

# Run the container
containers = Containers(image)
container = containers.create()
containers.run()
```

### Markdown & Zettelkasten

The `ures.markdown` module offers tools for working with Markdown files, including YAML front matter. The `Zettelkasten` class builds on this for note-taking with structured metadata.

- `MarkdownDocument`: Load, create, and manipulate Markdown files with front matter.
- `Zettelkasten`: Create Zettelkasten notes with required fields such as `title`, `type`, and `tags`.

### Data structures

The `ures.data_structure` module provides reusable data structures:

- `TreeNode`: A tree structure with methods for adding, removing, and traversing nodes.
- `BiDirectional`: A bi-directional linked list for forward and backward traversal.

### Core utilities

- `ures.files`: File system helpers (recursive file listing, filtering by name, listing directories, creating temp folders).
- `ures.timedate`: Time/date conversion helpers (datetimes and Unix timestamps to formatted strings, getting the current time).
- `ures.string`: String utilities such as Zettelkasten-style IDs, unique IDs, memory-size formatting, and capitalization.
- `ures.secrets`: `SecureKeyManager` for managing API keys and other secrets via different storage methods.
- `ures.network`: IP and subnet helpers (validation, containment checks, and IP generation from subnets).

### Literature tooling (optional)

If you use the literature search and citation modules, URes can also help with:

- **Literature search**: Configuration-driven multi-source paper search, with adapters and caching.
- **Literature citation**: Extractors and helpers to manage citation data and apply citation rules.

## Development

To contribute to URes, please follow these steps:

1.  Clone the repository:
    ```bash
    git clone https://github.com/stone-home/Python-URes.git
    ```
2.  Install the dependencies:
    ```bash
    cd Python-URes
    poetry install
    ```
3.  Run the tests:
    ```bash
    poetry run pytest
    ```

## License

This project is licensed under the MIT License – see the [LICENSE](https://github.com/stone-home/Python-URes/blob/main/LICENSE) file for details.
