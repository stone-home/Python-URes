"""URes: reusable utilities for research and development.

Public helpers live in the subpackages and modules imported from `ures.*`
(Docker, Markdown/Zettelkasten, data structures, literature tooling, memory
simulation, and core file/string/time/secrets/network helpers).

Examples:
    >>> from ures.string import unique_id
    >>> isinstance(unique_id(), str)
    True
"""
