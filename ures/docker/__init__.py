"""Docker image, container, orchestration, and cleanup helpers.

Examples:
    >>> from ures.docker import Image
    >>> image = Image("python", tag="3.12-slim")
    >>> image.get_fullname()
    'python:3.12-slim'
"""

from .conf import BuildConfig, RuntimeConfig
from .container import Container
from .containers import Containers
from .image import Image, ImageOrchestrator
from .cleanup import DockerCleanup


__all__ = [
    "DockerCleanup",
    "BuildConfig",
    "RuntimeConfig",
    "Image",
    "Container",
    "Containers",
    "ImageOrchestrator",
]
