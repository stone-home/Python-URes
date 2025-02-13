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
    def __init__(self, image: Image, client: Optional[docker.DockerClient] = None):
        self._image: Image = image
        self._client: docker.DockerClient = client or docker.from_env()
        self._container: Optional[DockerContainer] = None

    @property
    def image_name(self) -> str:
        return self._image.get_fullname()

    @property
    def is_created(self) -> bool:
        return not self._container is None

    @property
    def status(self):
        try:
            status = self._client.containers.get(self._container.id).status
        except docker.errors.NotFound:
            status = "removed"
        return status

    @property
    def exit_code(self):
        try:
            exit_code = self._client.containers.get(self._container.id).attrs["State"][
                "ExitCode"
            ]
        except docker.errors.NotFound:
            exit_code = None
        return exit_code

    @property
    def is_running(self):
        return self.status == "running"

    def _construct_build_params(self, config: RuntimeConfig) -> dict:
        _config: RuntimeConfig = config
        _params = {
            "image": _config.image_name,
            "auto_remove": _config.remove,
            "detach": _config.detach,
            # "user": f"{os.getuid()}:{os.getgid()}",
        }
        if _config.cpus is not None:
            _params["cpuset_cpus"] = _config.cpus
        if _config.gpus is not None:
            _params["device_requests"] = [
                docker.types.DeviceRequest(
                    driver=_config.gpu_driver,
                    device_ids=_config.gpus,
                    capabilities=[["gpu"]],
                )
            ]
        if _config.memory is not None:
            _params["mem_limit"] = _config.memory
        if _config.entrypoint is not None:
            _params["entrypoint"] = [
                str(_entrypoint) for _entrypoint in _config.entrypoint
            ]
        if _config.command is not None:
            _params["command"] = [str(_command) for _command in _config.command]
        if _config.volumes is not None:
            _params["volumes"] = _config.volumes
        if _config.env is not None:
            _params["environment"] = _config.env
        if _config.name is not None:
            _params["name"] = _config.name
        if _config.user is not None:
            _params["user"] = _config.user
        return _params

    def _create_subnet(self, config: RuntimeConfig) -> docker.models.networks.Network:
        submask_list = config.subnet_mask.split("/")
        assert is_valid_ip_netmask(ip=submask_list[0], netmask=submask_list[1])
        assert verify_ip_in_subnet(ip=config.subnet_gateway, subnet=config.subnet_mask)
        try:
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
        if config.subnet is not None:
            try:
                net = self._client.networks.get(config.subnet)
            except docker.errors.NotFound as e:
                logger.warning(f"Could not found network: {config.subnet} with msg {e}")
                net = self._create_subnet(config=config)
            finally:
                verify_ip_in_subnet(
                    ip=config.ipv4, subnet=net.attrs["IPAM"]["Config"][0]["Subnet"]
                )
                net.connect(contain, ipv4_address=config.ipv4)

    def create(self, config: RuntimeConfig, tag: Optional[str] = None):
        if config.image_name != self.image_name or tag != self._image.tag:
            config.image_name = self._image.get_fullname(tag=tag)
            logger.warning(f"The image name is updated to {config.image_name}")
        run_params = self._construct_build_params(config)
        logger.debug(f"Running Configuration: {json.dumps(run_params)}")
        _container = self._client.containers.create(**run_params)
        self._connect_to_network(_container, config)
        self._container = _container

    @check_instance_variable("_container")
    def stop(self):
        logger.debug("Stopping container")
        self._container.stop()

    @check_instance_variable("_container")
    def remove(self):
        logger.debug("Removing container")
        self._container.remove()
        self._container = None

    @check_instance_variable("_container")
    def logs(self):
        logger.debug("Get Logs from container")
        return self._container.logs()

    @check_instance_variable("_container")
    def wait(self):
        logger.debug("Waiting container")
        self._container.wait()

    def run(self):
        if self.is_created is True and self.is_running is False:
            self._container.start()
            logger.debug("Started container")
        else:
            if self.is_created is False:
                raise RuntimeError(f"Container has not been created: {self.image_name}")
            if self.is_running is True:
                raise RuntimeError(f"Container already running: {self.image_name}")
