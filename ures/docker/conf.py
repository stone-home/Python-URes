import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Tuple, Union
from ures.files import get_temp_folder

logger = logging.getLogger(__name__)


class BuildConfig(BaseModel):
    """
    BuildConfig defines the build parameters for constructing a Docker image.

    Attributes:
        base_image (str): Base image for the container. Default is "python:3.10-slim".
        platform (Optional[str]): Target platform in format os[/arch[/variant]].
        python_deps_manager (Optional[str]): Package manager for Python dependencies (e.g., pip, conda).
        python_dependencies (Optional[List[str]]): List of Python packages to install.
        sys_deps_manager (Optional[str]): Package manager for system dependencies (e.g., apt, yum, apk).
        sys_dependencies (Optional[List[str]]): List of system packages to install.
        labels (Optional[List[Tuple[str, str]]]): List of key-value tuples used as labels.
        uid (Optional[int]): User ID to use inside the container.
        user (Optional[str]): Username to use inside the container.
        entrypoint (Optional[List[str]]): Entrypoint command for the container.
        cmd (Optional[List[str]]): Default command to run in the container.
        environment (Optional[Dict[str, str]]): Environment variables for the container.
        copies (Optional[List[Dict[str, str]]]): File copy instructions (each dict should include "src" and "dest").
        context_dir (Union[str, Path]): Directory for the build context. Defaults to the current working directory.
        docker_filename (str): Filename for the Dockerfile. Defaults to "Dockerfile".

    Example:
        >>> config = BuildConfig()
        >>> config.base_image
        'python:3.10-slim'
    """

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
        description="Define which package manager to use for Python dependencies",
    )
    python_dependencies: Optional[List[str]] = Field(
        default=None,
        title="Python Dependencies",
        description="List of Python dependencies to be installed",
    )
    sys_deps_manager: Optional[str] = Field(
        default="apt",
        title="System Dependencies Manager, such as apt, yum, or apk",
        description="Define which package manager to use for system dependencies",
    )
    sys_dependencies: Optional[List[str]] = Field(
        default=None,
        title="System Dependencies",
        description="List of system dependencies to be installed",
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
    environment: Optional[Dict[str, str]] = Field(
        default=None,
        title="Environment",
        description="Environment variables for the container",
    )
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

    run_commands: Optional[List[str]] = Field(
        default=None,
        title="System Commands",
        description="Commands for running extra commands in the container, suck as installation of nightly pytorch",
    )

    def add_label(self, key: str, value: str):
        """
        Add a label to the build configuration.

        Args:
            key (str): The label key.
            value (str): The label value.

        Returns:
            None

        Example:
            >>> config = BuildConfig()
            >>> config.add_label("version", "1.0")
            >>> config.labels
            [("version", "1.0")]
        """
        logger.info(f"Adding label {key} with value {value}")
        if self.labels is None:
            self.labels = []
        self.labels.append((key, value))

    def add_copy(self, src: str, dest: str):
        """
        Add a file copy instruction to the build configuration.

        Args:
            src (str): Source file path.
            dest (str): Destination path inside the container.

        Returns:
            None

        Example:
            >>> config = BuildConfig()
            >>> config.add_copy("app.py", "/app/app.py")
            >>> config.copies
            [{"src": "app.py", "dest": "/app/app.py"}]
        """
        logger.info(f"Adding copy {src} to {dest}")
        if self.copies is None:
            self.copies = []
        self.copies.append({"src": src, "dest": dest})

    def add_environment(self, key: str, value: str):
        """
        Add an environment variable to the build configuration.

        Args:
            key (str): The environment variable name.
            value (str): The value for the environment variable.

        Returns:
            None

        Example:
            >>> config = BuildConfig()
            >>> config.add_environment("DEBUG", "true")
            >>> config.environment["DEBUG"]
            'true'
        """
        logger.info(f"Adding environment variable {key} with value {value}")
        if self.environment is None:
            self.environment = {}
        self.environment[key] = value

    def set_context_dir(self, context_dir: Union[str, Path]):
        """
        Set the build context directory.

        Args:
            context_dir (Union[str, Path]): The directory to be used as the build context.

        Returns:
            None

        Raises:
            ValueError: If the provided context directory does not exist or is not a directory.

        Example:
            >>> from pathlib import Path
            >>> config = BuildConfig()
            >>> temp_dir = Path("/tmp")
            >>> config.set_context_dir(temp_dir)
            >>> config.context_dir == temp_dir
            True
        """
        logger.info(f"Setting context directory to {context_dir}")
        if isinstance(context_dir, str):
            context_dir = Path(context_dir)
        if not context_dir.is_dir():
            raise ValueError(f"Context directory {context_dir} is not a directory")
        self.context_dir = context_dir

    def add_python_dependency(self, dependency: str):
        """
        Add a Python dependency to be installed in the image.

        Args:
            dependency (str): The Python package dependency (e.g., "flask==2.0.1").

        Returns:
            None

        Example:
            >>> config = BuildConfig()
            >>> config.add_python_dependency("flask")
            >>> "flask" in config.python_dependencies
            True
        """
        logger.info(f"Adding Python dependency: '{dependency}'")
        if self.python_dependencies is None:
            self.python_dependencies = []
        self.python_dependencies.append(dependency)

    def add_system_dependency(self, dependency: str):
        """
        Add a system dependency to be installed in the image.

        Args:
            dependency (str): The system package dependency (e.g., "curl").

        Returns:
            None

        Example:
            >>> config = BuildConfig()
            >>> config.add_system_dependency("curl")
            >>> "curl" in config.sys_dependencies
            True
        """
        logger.info(f"Adding system dependency: '{dependency}'")
        if self.sys_dependencies is None:
            self.sys_dependencies = []
        self.sys_dependencies.append(dependency)

    def add_run_command(self, command: str):
        """
        Add a command to be run during the build process.

        Args:
            command (str): The command to run (e.g., "apt-get update").

        Returns:
            None

        Example:
            >>> config = BuildConfig()
            >>> config.add_run_command("apt-get update")
            >>> "apt-get update" in config.run_commands
            True
        """
        logger.info(f"Adding run command: '{command}'")
        if self.run_commands is None:
            self.run_commands = []
        self.run_commands.append(command)

    def set_entrypoint(self, entrypoint: Union[str, List[str]]):
        """
        Set the entrypoint for the container.

        Args:
            entrypoint (Union[str, List[str]]): The entrypoint command(s). If a string is provided,
                                                it will be converted to a list.

        Returns:
            None

        Example:
            >>> config = BuildConfig()
            >>> config.set_entrypoint("python app.py")
            >>> config.entrypoint
            ["python app.py"]
        """
        if isinstance(entrypoint, str):
            entrypoint = [entrypoint]
        logger.info(f"Setting entrypoint to: '{entrypoint}'")
        self.entrypoint = entrypoint

    def set_cmd(self, cmd: Union[str, List[str]]):
        """
        Set the command for the container.

        Args:
            cmd (Union[str, List[str]]): The command(s) to run. If a string is provided,
                                         it will be converted to a list.

        Returns:
            None

        Example:
            >>> config = BuildConfig()
            >>> config.set_cmd("python -m myapp")
            >>> config.cmd
            ["python -m myapp"]
        """
        if isinstance(cmd, str):
            cmd = [cmd]
        logger.info(f"Setting command to: '{cmd}'")
        self.cmd = cmd


class RuntimeConfig(BaseModel):
    """
    RuntimeConfig defines the runtime parameters for running a Docker container.

    Attributes:
        image_name (str): Name of the Docker image in the format "image:tag".
        name (Optional[str]): Container name.
        platform (Optional[str]): Platform for the container.
        detach (bool): Whether to run the container in detached mode.
        user (Optional[str]): User under which to run the container.
        remove (bool): Whether to remove the container after it stops.
        cpus (Optional[int]): Number of CPUs to allocate.
        gpus (Optional[List[str]]): List of GPUs to allocate.
        gpu_driver (str): Driver to use for GPUs. Default is "nvidia".
        memory (Optional[str]): Memory limit for the container (e.g., "2g").
        entrypoint (Optional[List[Union[str, float, int, Path]]]): Entrypoint command(s).
        command (Optional[List[Union[int, float, str, Path]]]): Command(s) to run.
        env (Optional[Dict[str, str]]): Environment variables.
        volumes (Optional[Dict[str, Dict[str, str]]]): Volume mappings.
        subnet (Optional[str]): Subnet for container networking.
        ipv4 (Optional[str]): Specific IPv4 address for the container.
        subnet_mask (Optional[str]): Subnet mask (e.g., "172.17.0.0/16").
        subnet_gateway (Optional[str]): Subnet gateway.
        network_mode (Optional[str]): Docker network mode.
        out_dir (Path): Output directory for logs and cache.

    Example:
        >>> config = RuntimeConfig()
        >>> config.image_name
        'model-runner'
    """

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
    cpus: Optional[int] = Field(
        default=None,
        title="CPUs",
        description="Number of CPUs to allocate to the container. CPUs allowed (e.g., 0-3, 0,1)",
    )
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
    memory: Optional[str] = Field(
        default=None,
        title="Memory",
        description="Memory to allocate to the container with a units identification char (100000b, 1000k, 128m, 1g)",
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
        description="Directory for logs, cache, and other temporary files",
    )

    @property
    def log_dir(self) -> Path:
        """
        Get the log directory within the output directory. If it does not exist, it is created.

        Returns:
            Path: The path to the log directory.

        Example:
            >>> config = RuntimeConfig()
            >>> log_directory = config.log_dir
            >>> log_directory.exists()
            True
        """
        log_dir = self.out_dir.joinpath("log")
        if not log_dir.is_dir():
            log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    @property
    def cache(self) -> Path:
        """
        Get the cache directory within the output directory. If it does not exist, it is created.

        Returns:
            Path: The path to the cache directory.

        Example:
            >>> config = RuntimeConfig()
            >>> cache_directory = config.cache
            >>> cache_directory.exists()
            True
        """
        cache_dir = self.out_dir.joinpath("cache")
        if not cache_dir.is_dir():
            cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def add_volume(
        self,
        host_path: Union[str, Path],
        container_path: Union[str, Path],
        mode: str = "rw",
    ):
        """
        Add a volume mapping for the container.

        Args:
            host_path (Union[str, Path]): The path on the host machine.
            container_path (Union[str, Path]): The destination path inside the container.
            mode (str, optional): The mode for the volume mapping (e.g., "rw" or "ro"). Defaults to "rw".

        Returns:
            None

        Example:
            >>> config = RuntimeConfig()
            >>> config.add_volume("/host/data", "/container/data", mode="rw")
            >>> "/host/data" in config.volumes
            True
        """
        logger.info(f"Adding volume {host_path} to {container_path} with mode {mode}")
        if self.volumes is None:
            self.volumes = {}
        self.volumes[str(host_path)] = {"bind": str(container_path), "mode": mode}

    def add_env(self, key: str, value: str):
        """
        Add an environment variable for the container.

        Args:
            key (str): The environment variable name.
            value (str): The value for the environment variable.

        Returns:
            None

        Example:
            >>> config = RuntimeConfig()
            >>> config.add_env("DEBUG", "1")
            >>> config.env["DEBUG"]
            '1'
        """
        logger.info(f"Adding environment variable {key} with value {value}")
        if self.env is None:
            self.env = {}
        self.env[key] = value
