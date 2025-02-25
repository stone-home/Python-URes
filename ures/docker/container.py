import logging
import docker
import docker.constants
import docker.models.containers
import docker.models.networks
import docker.types
import docker.errors
import json
from docker.models.containers import Container as DockerContainer
from typing import Optional
from ures.network import verify_ip_in_subnet, is_valid_ip_netmask
from ures.tools.decorator import check_instance_variable
from .image import Image
from .conf import RuntimeConfig

logger = logging.getLogger(__name__)


class Container:
    """
    A class to manage a Docker container.

    This class provides methods to create a container with specified runtime configurations,
    manage network connections, and control the container's lifecycle (start, stop, remove,
    logs, wait).

    Attributes:
        _image (Image): The Docker image object to be used.
        _client (docker.DockerClient): The Docker client instance.
        _container (Optional[DockerContainer]): The underlying Docker container object.
    """

    def __init__(self, image: Image, client: Optional[docker.DockerClient] = None):
        """
        Initialize a Container instance.

        Args:
            image (Image): The Image object that provides the Docker image details.
            client (Optional[docker.DockerClient]): An optional Docker client instance. If not provided,
                docker.from_env() is used.

        Example:
            >>> from ures.docker.image import Image
            >>> img = Image("myapp")
            >>> container = Container(img)
        """
        self._image: Image = image
        self._client: docker.DockerClient = client or docker.from_env()
        self._container: Optional[DockerContainer] = None

    @property
    def image_name(self) -> str:
        """
        Retrieve the full image name (including tag).

        Returns:
            str: The full image name.

        Example:
            >>> container.image_name
            'myapp:latest'
        """
        return self._image.get_fullname()

    @property
    def is_created(self) -> bool:
        """
        Check whether the container has been created.

        Returns:
            bool: True if the container exists, False otherwise.

        Example:
            >>> container.is_created
            False
        """
        return self._container is not None

    @property
    def status(self) -> str:
        """
        Get the current status of the container.

        Returns:
            str: The container status. If not found, returns "removed".

        Example:
            >>> status = container.status
            >>> status in ["created", "running", "exited", "removed"]
            True
        """
        try:
            status = self._client.containers.get(self._container.id).status
        except docker.errors.NotFound:
            status = "removed"
        return status

    @property
    def exit_code(self):
        """
        Retrieve the exit code of the container's last run.

        Returns:
            int or None: The exit code if available, otherwise None.

        Example:
            >>> code = container.exit_code
            >>> isinstance(code, int) or code is None
            True
        """
        try:
            exit_code = self._client.containers.get(self._container.id).attrs["State"][
                "ExitCode"
            ]
        except docker.errors.NotFound:
            exit_code = None
        return exit_code

    @property
    def is_running(self) -> bool:
        """
        Check if the container is currently running.

        Returns:
            bool: True if running, False otherwise.

        Example:
            >>> container.is_running
            True
        """
        return self.status == "running"

    def _construct_build_params(self, config: RuntimeConfig) -> dict:
        """
        Construct runtime parameters for creating the container from the given configuration.

        Args:
            config (RuntimeConfig): The runtime configuration for the container.

        Returns:
            dict: A dictionary of parameters to be passed to Docker for container creation.

        Example:
            >>> params = container._construct_build_params(config)
            >>> isinstance(params, dict)
            True
        """
        _config: RuntimeConfig = config
        _params = {
            "image": _config.image_name,
            "auto_remove": _config.remove,
            "detach": _config.detach,
        }
        logger.debug(f"Creating container with configuration: {json.dumps(_params)}")
        if _config.cpus is not None:
            logger.debug(f"Setting cpuset_cpus to {_config.cpus}")
            _params["cpuset_cpus"] = _config.cpus
        if _config.gpus is not None:
            logger.debug(f"Setting device_requests to {_config.gpus}")
            _params["device_requests"] = [
                docker.types.DeviceRequest(
                    driver=_config.gpu_driver,
                    device_ids=_config.gpus,
                    capabilities=[["gpu"]],
                )
            ]
        if _config.memory is not None:
            logger.debug(f"Setting mem_limit to {_config.memory}")
            _params["mem_limit"] = _config.memory
        if _config.entrypoint is not None:
            logger.debug(f"Setting entrypoint to {_config.entrypoint}")
            _params["entrypoint"] = [str(entry) for entry in _config.entrypoint]
        if _config.command is not None:
            logger.debug(f"Setting command to {_config.command}")
            _params["command"] = [str(cmd) for cmd in _config.command]
        if _config.volumes is not None:
            logger.debug(f"Setting volumes to {_config.volumes}")
            _params["volumes"] = _config.volumes
        if _config.env is not None:
            logger.debug(f"Setting environment to {_config.env}")
            _params["environment"] = _config.env
        if _config.name is not None:
            logger.debug(f"Setting name to {_config.name}")
            _params["name"] = _config.name
        if _config.user is not None:
            logger.debug(f"Setting user to {_config.user}")
            _params["user"] = _config.user
        return _params

    def _create_subnet(self, config: RuntimeConfig) -> docker.models.networks.Network:
        """
        Create a new Docker network (subnet) based on the provided configuration.

        Args:
            config (RuntimeConfig): The runtime configuration containing subnet parameters.

        Returns:
            docker.models.networks.Network: The created Docker network.

        Raises:
            RuntimeError: If the subnet creation fails.

        Example:
            >>> net = container._create_subnet(config)
            >>> net.name == config.subnet
            True
        """
        submask_list = config.subnet_mask.split("/")
        assert is_valid_ip_netmask(ip=submask_list[0], netmask=submask_list[1])
        assert verify_ip_in_subnet(ip=config.subnet_gateway, subnet=config.subnet_mask)
        try:
            logger.debug(
                f"Creating IPAM for docker network with mask {config.subnet_mask} and gateway {config.subnet_gateway}"
            )
            logger.debug(
                f"Create network with name {config.subnet}, driver {config.network_mode}"
            )
            ipam_pool = docker.types.IPAMPool(
                subnet=config.subnet_mask, gateway=config.subnet_gateway
            )
            ipam_config = docker.types.IPAMConfig(pool_configs=[ipam_pool])
            network = self._client.networks.create(
                **{
                    "name": config.subnet,
                    "driver": config.network_mode,
                    "ipam": ipam_config,
                }
            )
        except docker.errors.APIError as e:
            raise RuntimeError(f"Failed to create subnet {config.subnet}") from e
        else:
            return network

    def _connect_to_network(self, contain: DockerContainer, config: RuntimeConfig):
        """
        Connect the container to a Docker network based on the provided configuration.

        Args:
            contain (DockerContainer): The Docker container object to connect.
            config (RuntimeConfig): The runtime configuration with network details.

        Returns:
            None

        Example:
            >>> container._connect_to_network(docker_container, config)
        """
        if config.subnet is not None:
            try:
                logger.info(f"Connecting container to network: {config.subnet}")
                net = self._client.networks.get(config.subnet)
            except docker.errors.NotFound as e:
                logger.warning(f"Could not find network: {config.subnet} with msg {e}")
                net = self._create_subnet(config=config)
            finally:
                verify_ip_in_subnet(
                    ip=config.ipv4, subnet=net.attrs["IPAM"]["Config"][0]["Subnet"]
                )
                logger.debug(f"Connecting container to network with IP {config.ipv4}")
                net.connect(contain, ipv4_address=config.ipv4)

    def create(self, config: RuntimeConfig, tag: Optional[str] = None):
        """
        Create a Docker container using the provided runtime configuration.

        Args:
            config (RuntimeConfig): The runtime configuration for the container.
            tag (Optional[str]): An optional image tag override. Defaults to None.

        Returns:
            None

        Example:
            >>> container.create(runtime_config)
            >>> container.is_created
            True
        """
        if config.image_name != self.image_name or tag != self._image.tag:
            config.image_name = self._image.get_fullname(tag=tag)
            logger.warning(f"The image name is updated to {config.image_name}")
        run_params = self._construct_build_params(config)
        logger.debug(
            f"Create {self.image_name} with running Configuration: {json.dumps(run_params)}"
        )
        _container = self._client.containers.create(**run_params)
        self._connect_to_network(_container, config)
        self._container = _container

    @check_instance_variable("_container")
    def stop(self):
        """
        Stop the running container.

        Returns:
            None

        Example:
            >>> container.stop()
        """
        logger.debug(f"Stopping container: {self.image_name}")
        self._container.stop()

    @check_instance_variable("_container")
    def remove(self):
        """
        Remove the container.

        Returns:
            None

        Example:
            >>> container.remove()
        """
        logger.debug(f"Removing container: {self.image_name}")
        self._container.remove()
        self._container = None

    @check_instance_variable("_container")
    def logs(self):
        """
        Retrieve logs from the container.

        Returns:
            bytes: The log output from the container.

        Example:
            >>> logs = container.logs()
            >>> isinstance(logs, bytes)
            True
        """
        logger.debug(f"Retrieving logs from container: {self.image_name}")
        return self._container.logs()

    @check_instance_variable("_container")
    def wait(self):
        """
        Wait for the container to finish execution.

        Returns:
            dict: The container's exit information.

        Example:
            >>> exit_info = container.wait()
            >>> "StatusCode" in exit_info
            True
        """
        logger.debug(f"Waiting for container to finish: {self.image_name}")
        self._container.wait()

    def run(self):
        """
        Start the container if it has been created and is not already running.

        Raises:
            RuntimeError: If the container has not been created or is already running.

        Returns:
            None

        Example:
            >>> container.run()
            >>> container.is_running
            True
        """
        if self.is_created is True and self.is_running is False:
            self._container.start()
            logger.debug(f"Container started: {self.image_name}")
        else:
            if self.is_created is False:
                raise RuntimeError(f"Container has not been created: {self.image_name}")
            if self.is_running is True:
                raise RuntimeError(f"Container already running: {self.image_name}")
