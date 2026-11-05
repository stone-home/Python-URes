# Utility for Research (URes)

[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Super-Linter](https://github.com/stone-home/Python-URes/actions/workflows/linter.yaml/badge.svg)](https://github.com/stone-home/Python-URes/actions/workflows/linter.yaml)
[![Trufflehog](https://github.com/stone-home/Python-URes/actions/workflows/secret-check.yaml/badge.svg)](https://github.com/stone-home/Python-URes/actions/workflows/secret-check.yaml)
[![Code Testing](https://github.com/stone-home/Python-URes/actions/workflows/test.yaml/badge.svg)](https://github.com/stone-home/Python-URes/actions/workflows/test.yaml)

## Installation

```bash
pip install ures
```

Need 1Password? Install the extra:

```bash
pip install 'ures[onepassword]'
```

Poetry: `ures = { version = "^3.3.0", extras = ["onepassword"] }`

Since 3.3.0 the 1Password SDK is optional, because it does not install on Alpine Linux.
If your project stores keys in 1Password, add the extra when you upgrade.
Without it, 1Password keys come back as `None`.
