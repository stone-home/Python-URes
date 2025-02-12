import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Tuple, Union


logger = logging.getLogger(__name__)


class BuildConfig(BaseModel):
    base_image: str = Field(
        default="python:3.10-slim",
        title="Base Image",
        description="Base image for the container",
    )
    platform: Optional[str] = Field(
        default=None, title="Platform", description="Platform for the base image in format os[/arch[/variant]]"
    )
    python_deps_manager: Optional[str] = Field(
        default="pip",
        title="Python Dependencies Manager, such as pip or conda",
        description="To define which package manager to use for python dependencies",
    )
    python_dependencies: Optional[List[str]] = Field(
        default=None,
        title="Python Dependencies",
        description="Python dependencies to be installed",
    )
    sys_deps_manager: Optional[str] = Field(
        default="apt",
        title="System Dependencies Manager, such as apt, yum, or apk",
        description="To define which package manager to use for system dependencies",
    )
    sys_dependencies: Optional[List[str]] = Field(
        default=None,
        title="System Dependencies",
        description="System dependencies to be installed",
    )
    labels: Optional[List[Tuple[str, str]]] = Field(
        default=None, title="Labels", description="Labels for the image"
    )
    uid: Optional[int] = Field(
        default=None, title="User ID", description="User ID for the container"
    )
    user: Optional[str] = Field(
        default=None, title="User", description="User for the container"
    )
    entrypoint: Optional[List[str]] = Field(
        default=None, title="Entrypoint", description="Entrypoint for the container"
    )
    cmd: Optional[List[str]] = Field(
        default=None, title="Command", description="Command for the container"
    )
    # The data structure for environment is a dictionary with the keys being the environment variable names
    environment: Optional[Dict[str, str]] = Field(
        default=None,
        title="Environment",
        description="Environment variables for the container",
    )
    # The data structure for copies is a list of dictionaries with the keys "src", "dest", and "mode"
    copies: Optional[List[Dict[str, str]]] = Field(
        default=None, title="Copies", description="Files to be copied to the container"
    )
    context_dir: Union[str, Path] = Field(
        default=Path().cwd(),
        title="Context Directory",
        description="Directory for the build context",
    )

    def add_label(self, key: str, value: str):
        logger.info(f"Adding label {key} with value {value}")
        if self.labels is None:
            self.labels = []
        self.labels.append((key, value))

    def add_copy(self, src: str, dest: str):
        logger.info(f"Adding copy {src} to {dest}")
        if self.copies is None:
            self.copies = []
        self.copies.append({"src": src, "dest": dest})

    def add_environment(self, key: str, value: str):
        logger.info(f"Adding environment variable {key} with value {value}")
        if self.environment is None:
            self.environment = {}
        self.environment[key] = value

    def set_context_dir(self, context_dir: Union[str, Path]):
        logger.info(f"Setting context directory to {context_dir}")
        if isinstance(context_dir, str):
            context_dir = Path(context_dir)
        if not context_dir.is_dir():
            raise ValueError(f"Context directory {context_dir} is not a directory")
        self.context_dir = context_dir