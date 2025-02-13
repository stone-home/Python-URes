import logging
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Tuple, Union
from ures.files import get_temp_folder


logger = logging.getLogger(__name__)


class BuildConfig(BaseModel):
    base_image: str = Field(
        default="python:3.10-slim",
        title="Base Image",
        description="Base image for the container",
    )
    platform: Optional[str] = Field(
        default=None,
        title="Platform",
        description="Platform for the base image in format os[/arch[/variant]]",
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
    docker_filename: str = Field(
        default="Dockerfile",
        title="Dockerfile",
        description="Name of the Dockerfile",
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

    def add_python_dependency(self, dependency: str):
        """Adds a Python dependency."""
        logger.info(f"Adding Python dependency: '{dependency}'")
        if self.python_dependencies is None:
            self.python_dependencies = []
        self.python_dependencies.append(dependency)

    def add_system_dependency(self, dependency: str):
        """Adds a system dependency."""
        logger.info(f"Adding system dependency: '{dependency}'")
        if self.sys_dependencies is None:
            self.sys_dependencies = []
        self.sys_dependencies.append(dependency)

    def set_entrypoint(self, entrypoint: Union[str, List[str]]):
        """Sets the entrypoint for the container."""
        if isinstance(entrypoint, str):
            entrypoint = [entrypoint]
        logger.info(f"Setting entrypoint to: '{entrypoint}'")
        self.entrypoint = entrypoint

    def set_cmd(self, cmd: Union[str, List[str]]):
        """Sets the command for the container."""
        if isinstance(cmd, str):
            cmd = [cmd]
        logger.info(f"Setting command to: '{cmd}'")
        self.cmd = cmd


class RuntimeConfig(BaseModel):
    image_name: str = Field(
        default="model-runner",
        title="Image Name",
        description="Name of the image in form of image:tag",
    )
    name: Optional[str] = Field(
        default=None, title="Name", description="Name of the container"
    )
    platform: Optional[str] = Field(
        default=None, title="Platform", description="Platform for the base image"
    )
    detach: bool = Field(
        default=True, title="Detach", description="Run the container in detached mode"
    )
    user: Optional[str] = Field(
        default=None, title="User", description="User for the container"
    )
    remove: bool = Field(
        default=False,
        title="Remove",
        description="Remove the container after it is stopped",
    )
    # for parameter cpus, the type is Optional[int] because the number of CPUs can be None
    cpus: Optional[int] = Field(
        default=None,
        title="CPUs",
        description="Number of CPUs to allocate to the container. "
        "CPUs in which to allow execution (0-3, 0,1)",
    )
    # for parameter gpus, the type is List[str] because the GPU IDs are strings, for example ["0", "1"]
    gpus: Optional[List[str]] = Field(
        default=None,
        title="GPUs",
        description="List of GPUs to allocate to the container",
    )
    gpu_driver: str = Field(
        default="nvidia",
        title="GPU Driver",
        description="The driver to use for the GPUs, defaults to nvidia",
    )
    # for parameter memory, the type is Optional[str] because the memory can be None, for example "2g"
    memory: Optional[str] = Field(
        default=None, title="Memory", description="Memory to allocate to the container"
    )
    entrypoint: Optional[List[Union[str, float, int, Path]]] = Field(
        default=None, title="Entrypoint", description="Entrypoint for the container"
    )
    command: Optional[List[Union[int, float, str, Path]]] = Field(
        default=None, title="Command", description="Command for the container"
    )
    env: Optional[Dict[str, str]] = Field(
        default=None,
        title="Environment",
        description="Environment variables for the container",
    )
    volumes: Optional[Dict[str, Dict[str, str]]] = Field(
        default=None, title="Volumes", description="Volumes to mount to the container"
    )
    subnet: Optional[str] = Field(
        default=None, title="Subnet", description="Subnet for the container"
    )
    ipv4: Optional[str] = Field(
        default=None, title="IPv4", description="Specified IPv4 for the container"
    )
    subnet_mask: Optional[str] = Field(
        default="172.17.0.0/16", title="IPv4", description="IPv4 Mask for Subnet"
    )
    subnet_gateway: Optional[str] = Field(
        default="172.17.0.1",
        title="Subnet Gateway",
        description="Subnet Gateway for the container",
    )
    network_mode: Optional[str] = Field(
        default="bridge",
        title="Network Mode",
        description="Network mode for the container",
    )
    out_dir: Path = Field(
        default=Path(get_temp_folder()),
        title="Cache Directory",
        description="Directory for the cache",
    )

    @property
    def log_dir(self) -> Path:
        log_dir = self.out_dir.joinpath("log")
        if log_dir.is_dir() is False:
            log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    @property
    def cache(self) -> Path:
        cache_dir = self.out_dir.joinpath("cache")
        if cache_dir.is_dir() is False:
            cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def add_volume(
        self,
        host_path: Union[str, Path],
        container_path: Union[str, Path],
        mode: str = "rw",
    ):
        logger.info(f"Adding volume {host_path} to {container_path} with mode {mode}")
        if self.volumes is None:
            self.volumes = {}
        self.volumes[str(host_path)] = {"bind": str(container_path), "mode": mode}

    def add_env(self, key: str, value: str):
        logger.info(f"Adding environment variable {key} with value {value}")
        if self.env is None:
            self.env = {}
        self.env[key] = value
