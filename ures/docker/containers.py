import docker
from pathlib import Path
from typing import Optional, Dict, Union, List, Tuple
from ures.string import unique_id
from ures.files import get_temp_dir_with_specific_path
from .runtime import Runtime, SimpleRuntime
from .image import Image
from .container import Container
from .conf import RuntimeConfig


class Containers:
    """
    Manages the creation and lifecycle of Docker container instances derived from a given Docker image.

    This class provides methods to create new container instances using a default or customized runtime
    configuration, run those containers using a specified runtime (default is SimpleRuntime), and then stop
    and remove them while maintaining a history of container records and their configurations.

    Args:
        image (Image): An instance of the Image class representing the Docker image to be used.
        client (Optional[docker.DockerClient], optional): A Docker client instance. Defaults to docker.from_env().
        runtime (type(Runtime), optional): The runtime class used to manage container operations.
            Must be a subclass of Runtime. Defaults to SimpleRuntime.

    Attributes:
        _client (docker.DockerClient): The Docker client instance.
        _image (Image): The Docker image used for creating containers.
        _runtime_history (Dict[str, Dict[str, Union[Container, str]]]): A dictionary storing records of created
            container instances and their associated runtime configurations, keyed by a unique container name.
        _runtime (type(Runtime)): The runtime class used for managing container lifecycle operations.

    Methods:
        get_container(new: bool = False) -> List[Tuple[str, Dict[str, Union[Container, str]]]]:
            Retrieve container records from the history. If `new` is True, returns only records whose container
            status is "created".
        _default_config() -> RuntimeConfig:
            Generates a default RuntimeConfig based on the image and a generated unique container name.
        _construct_config(**kwargs) -> RuntimeConfig:
            Constructs a RuntimeConfig by updating the default configuration with provided keyword arguments.
        _add_record(container: Container, config: RuntimeConfig) -> None:
            Adds a record for a newly created container and its configuration to the runtime history.
        create(**kwargs) -> Container:
            Creates a new container using the provided configuration keyword arguments, records it in the history,
            and returns the container.
        run() -> List[Tuple[str, Dict[str, Union[Container, str]]]]:
            Runs all newly created containers using the specified runtime and returns their records.
        stop(log_dir: Optional[Union[str, Path]] = None) -> List[Tuple[str, Dict[str, Union[Container, str]]]]:
            Stops all managed containers, collects their logs (saving them to the specified directory), removes them,
            and returns the updated container records.

    Example:
        >>> from ures.docker.image import Image
        >>> from ures.docker.containers import Containers
        >>> from ures.docker.conf import RuntimeConfig
        >>> img = Image("myapp", tag="latest")
        >>> containers_manager = Containers(image=img)
        >>> container = containers_manager.create(name="myapp-instance")
        >>> containers_manager.run()
        >>> containers_manager.stop("/tmp/container_logs")
    """

    def __init__(
        self,
        image: Image,
        client: Optional[docker.DockerClient] = None,
        runtime: type(Runtime) = SimpleRuntime,
    ):
        # Ensure that the image is an instance of the Image class
        assert isinstance(image, Image)
        # Ensure that the provided runtime is a subclass of Runtime
        assert isinstance(runtime, type(Runtime))
        self._client = client or docker.from_env()
        self._image = image
        self._runtime_history: Dict[str, Dict[str, Union[Container, str]]] = {}
        self._runtime: type(Runtime) = runtime

    @property
    def image(self) -> str:
        """
        Returns the full image name including the tag.

        Returns:
            str: The full image name.

        Example:
            >>> containers_manager.image
            'myapp:latest'
        """
        return self._image.get_fullname()

    @property
    def name(self):
        """
        Generates a unique container name.

        Returns:
            str: A unique container name in the format "{image.name}-instance-{unique_id}".

        Example:
            >>> containers_manager.name
            'myapp-instance-abc123def4'
        """
        return f"{self._image.name}-instance-{unique_id()[:10]}"

    @property
    def history(self) -> Dict[str, Dict[str, Union[Container, str]]]:
        """
        Retrieves the history of container records.

        Returns:
            Dict[str, Dict[str, Union[Container, str]]]: A dictionary of container records keyed by unique names.

        Example:
            >>> containers_manager.history
            {'myapp-instance-abc123def4': {'container': <Container object>, 'config': <RuntimeConfig>}}
        """
        return self._runtime_history

    def get_container(
        self, new: bool = False
    ) -> List[Tuple[str, Dict[str, Union[Container, str]]]]:
        """
        Retrieve container records from the history.

        Args:
            new (bool, optional): If True, only return container records whose container status is "created".
                Defaults to False.

        Returns:
            List[Tuple[str, Dict[str, Union[Container, str]]]]: A list of tuples containing the container's unique name
            and its record.

        Example:
            >>> records = containers_manager.get_container(new=True)
        """
        if new:
            containers = list(
                filter(
                    lambda x: x[1]["container"].status == "created",
                    self._runtime_history.items(),
                )
            )
        else:
            containers = list(self._runtime_history.items())
        return containers

    def _default_config(self) -> RuntimeConfig:
        """
        Generates a default runtime configuration for container creation.

        Returns:
            RuntimeConfig: A default configuration with preset values based on the image and a unique name.

        Example:
            >>> config = containers_manager._default_config()
        """
        return RuntimeConfig(
            image_name=self.image, name=self.name, detach=True, remove=False
        )

    def _construct_config(self, **kwargs) -> RuntimeConfig:
        """
        Constructs a runtime configuration by updating the default configuration with additional parameters.

        Keyword Args:
            Arbitrary keyword arguments to override default configuration values.

        Returns:
            RuntimeConfig: The constructed configuration.

        Example:
            >>> config = containers_manager._construct_config(name="custom-instance")
        """
        _conf = self._default_config()
        kwargs.pop("image_name", None)
        _conf = _conf.model_copy(update=kwargs)
        return _conf

    def _add_record(self, container: Container, config: RuntimeConfig):
        """
        Adds a container record to the runtime history.

        Args:
            container (Container): The created container instance.
            config (RuntimeConfig): The runtime configuration used for creating the container.

        Returns:
            None

        Example:
            >>> containers_manager._add_record(container, config)
        """
        self._runtime_history[config.name] = {
            "container": container,
            "config": config,
        }

    def create(self, **kwargs) -> Container:
        """
        Create a new container instance using the specified configuration.

        Keyword Args:
            Arbitrary keyword arguments to override the default runtime configuration (e.g., name, detach).

        Returns:
            Container: The newly created container instance.

        Example:
            >>> container = containers_manager.create(name="instance1")
        """
        _container = Container(image=self._image, client=self._client)
        _conf = self._construct_config(**kwargs)
        _container.create(config=_conf, tag=None)
        self._add_record(container=_container, config=_conf)
        return _container

    def run(self) -> List[Tuple[str, Dict[str, Union[Container, str]]]]:
        """
        Run all newly created containers using the specified runtime.

        Returns:
            List[Tuple[str, Dict[str, Union[Container, str]]]]: The list of container records that were run.

        Example:
            >>> records = containers_manager.run()
        """
        new_containers = self.get_container(new=True)
        containers = [_c[1]["container"] for _c in new_containers]
        _runtime = self._runtime(containers=containers)
        _runtime.run()
        return new_containers

    def stop(
        self, log_dir: Optional[Union[str, Path]] = None
    ) -> List[Tuple[str, Dict[str, Union[Container, str]]]]:
        """
        Stop all managed containers, collect their logs (saving them to the specified directory), and remove them.

        Args:
            log_dir (Optional[Union[str, Path]], optional): The directory where container logs will be saved.
                If not provided, a default temporary directory is used.

        Returns:
            List[Tuple[str, Dict[str, Union[Container, str]]]]: The updated container records after stopping and removal.

        Example:
            >>> records = containers_manager.stop("/tmp/container_logs")
        """
        log_dir = log_dir or get_temp_dir_with_specific_path("container-logs")
        log_dir = Path(log_dir)
        containers = [_c[1]["container"] for _c in self.get_container()]
        _runtime = self._runtime(containers=containers)
        _runtime.stop()
        _runtime.logs(output_dir=log_dir)
        _runtime.remove()
        return self.get_container()
