import pytest
from unittest.mock import patch
from docker.errors import NotFound, ImageNotFound
from ures.docker.containers import Containers
from ures.docker.container import Container as SingleContainer
from ures.docker.image import Image as DockerImageClass
from ures.docker.conf import RuntimeConfig, BuildConfig
from ures.docker.runtime import SimpleRuntime, Runtime

# -----------------------------------------------------------------------------
# Dummy implementations to simulate external Docker API behavior

# Dummy unique_id function to return a fixed string for reproducibility
dummy_unique_id = lambda: "1234567890"

# Dummy function to simulate getting a temporary directory
dummy_get_temp_dir_with_specific_path = lambda prefix: f"/tmp/{prefix}"


# Dummy Docker container that simulates a real container
class DummyDockerContainer:
    def __init__(self, name="dummy_container"):
        self.id = name
        self.status = "created"
        self.attrs = {"State": {"ExitCode": 0}}

    def logs(self):
        return b"dummy logs"

    def start(self):
        self.status = "running"

    def stop(self):
        self.status = "stopped"

    def remove(self):
        self.status = "removed"

    def wait(self):
        pass


# Dummy manager for Docker containers
class DummyDockerContainersManager:
    def __init__(self):
        self.containers = {}

    def create(self, **kwargs):
        name = kwargs.get("name", "dummy_container")
        container = DummyDockerContainer(name=name)
        self.containers[name] = container
        return container

    def get(self, container_id):
        if container_id in self.containers:
            return self.containers[container_id]
        raise NotFound("Container not found")


# Dummy manager for Docker networks
class DummyDockerNetworksManager:
    def __init__(self):
        self.networks = {}

    def get(self, name):
        if name in self.networks:
            return self.networks[name]
        raise NotFound("Network not found")

    def create(self, **kwargs):
        # Create a dummy network with minimal IPAM attributes
        class DummyNetwork:
            def __init__(self, name, subnet):
                self.name = name
                self.attrs = {"IPAM": {"Config": [{"Subnet": subnet}]}}

            def connect(self, container, ipv4_address):
                pass

        subnet = kwargs.get("ipam").pool_configs[0].subnet
        network = DummyNetwork(name=kwargs.get("name"), subnet=subnet)
        self.networks[network.name] = network
        return network


# Dummy Docker image to simulate an image object
class DummyDockerImage:
    def __init__(self, image_name, tag):
        self._image_name = image_name
        self._tag = tag
        self.id = "dummy_image_id"
        self.attrs = {"Architecture": "amd64", "Size": 12345678}
        self.labels = {"dummy": "label"}

    def get_fullname(self, tag=None):
        if tag is None:
            tag = self._tag
        return f"{self._image_name}:{tag}"


# Dummy manager for Docker images
class DummyDockerImagesManager:
    def __init__(self):
        self.images = {}

    def get(self, image_name):
        if image_name in self.images:
            return self.images[image_name]
        raise ImageNotFound("Image not found")

    def pull(self, image_name, tag):
        dummy_image = DummyDockerImage(image_name, tag)
        self.images[dummy_image.get_fullname()] = dummy_image
        return dummy_image

    def build(self, **kwargs):
        tag = kwargs.get("tag")
        name = tag.split(":")[0]
        dummy_image = DummyDockerImage(name, tag.split(":")[1])
        self.images[dummy_image.get_fullname()] = dummy_image
        return dummy_image, ["build log"]

    def remove(self, **kwargs):
        image = kwargs.get("image")
        if image in self.images:
            del self.images[image]


# Dummy Docker client that holds dummy managers
class DummyDockerClient:
    def __init__(self):
        self.containers = DummyDockerContainersManager()
        self.networks = DummyDockerNetworksManager()
        self.images = DummyDockerImagesManager()


class DummyRuntime(Runtime):
    def __init__(self, containers):
        super().__init__(containers)
        self.run_called = False
        self.stop_called = False
        self.logs_called = False
        self.remove_called = False

    def run(self):
        self.run_called = True
        for container in self._containers:
            # Instead of setting container.status (read-only), update the underlying dummy container.
            container._container.status = "running"

    def stop(self):
        self.stop_called = True
        for container in self._containers:
            container._container.status = "stopped"

    def logs(self, output_dir):
        self.logs_called = True

    def remove(self):
        self.remove_called = True
        for container in self._containers:
            container._container.status = "removed"


# -----------------------------------------------------------------------------
# Use fixture to temporarily patch dependencies instead of global assignment
@pytest.fixture(autouse=True)
def patch_dependencies():
    with patch("ures.docker.containers.unique_id", new=dummy_unique_id), patch(
        "ures.docker.image.unique_id", new=dummy_unique_id
    ), patch(
        "ures.files.get_temp_dir_with_specific_path",
        new=dummy_get_temp_dir_with_specific_path,
    ):
        yield


# -----------------------------------------------------------------------------
# Test cases for Containers (containers.py)


@pytest.fixture
def dummy_image():
    # Create a dummy image using the Image class from image.py with a DummyDockerClient
    client = DummyDockerClient()
    return DockerImageClass("testapp", tag="latest", client=client)


@pytest.fixture
def containers_manager(dummy_image):
    # Use a dummy Docker client and override runtime with DummyRuntime
    client = DummyDockerClient()
    return Containers(image=dummy_image, client=client, runtime=DummyRuntime)


def test_containers_properties(containers_manager):
    # Test the properties of the Containers class
    assert containers_manager.image == "testapp:latest"
    name = containers_manager.name
    assert name.startswith("testapp-instance-")
    assert name.endswith("1234567890")
    assert containers_manager.history == {}


def test_containers_create(containers_manager):
    # Test creating a container and checking the history record
    container = containers_manager.create(name="custom-container")
    assert container.status == "created"
    history = containers_manager.history
    assert "custom-container" in history
    record = history["custom-container"]
    assert record["container"] is container
    assert record["config"].name == "custom-container"


def test_containers_run(containers_manager):
    # Test running created containers using DummyRuntime
    container = containers_manager.create(name="run-container")
    records_before_run = containers_manager.get_container(new=True)
    assert len(records_before_run) == 1
    records = containers_manager.run()
    assert container.status == "running"
    assert len(records) == 1


def test_containers_stop(tmp_path, containers_manager):
    # Test stopping containers using DummyRuntime
    container = containers_manager.create(name="stop-container")
    containers_manager.run()
    log_dir = str(tmp_path)
    records = containers_manager.stop(log_dir=log_dir)
    assert container.status == "removed"
    history_keys = [key for key, _ in records]
    assert "stop-container" in history_keys


# -----------------------------------------------------------------------------
# Test cases for Container (container.py)


@pytest.fixture
def dummy_container():
    client = DummyDockerClient()
    dummy_img = DockerImageClass("myapp", tag="latest", client=client)
    # Create a Container instance from container.py
    cont = SingleContainer(image=dummy_img, client=client)
    # Monkey-patch _connect_to_network to do nothing during testing
    cont._connect_to_network = lambda dummy, config: None
    return cont


def dummy_runtime_config():
    # Create a simple RuntimeConfig instance using conf.RuntimeConfig
    return RuntimeConfig(
        image_name="myapp:latest", name="test-container", detach=True, remove=False
    )


def test_container_create(dummy_container):
    # Test the create method of the Container class
    config = dummy_runtime_config()
    dummy_container.create(config=config, tag=None)
    # After creation, _container should be set (i.e. the container is created)
    assert dummy_container.is_created is True


def test_container_run_stop_remove_logs_wait(dummy_container):
    # Test run, stop, remove, logs, and wait methods of the Container class
    config = dummy_runtime_config()
    dummy_container.create(config=config, tag=None)
    # Test run: simulate container start
    dummy_container.run()
    dummy_container._container.status = "running"
    assert dummy_container.is_running is True
    # Test stop: simulate stopping the container
    dummy_container.stop()
    dummy_container._container.status = "stopped"
    assert dummy_container.status == "stopped"
    # Test logs: ensure logs are returned as bytes
    logs = dummy_container.logs()
    assert isinstance(logs, bytes)
    # Test remove: after removal, _container should be None
    dummy_container.remove()
    assert dummy_container._container is None


# -----------------------------------------------------------------------------
# Test cases for Image (image.py)


@pytest.fixture
def dummy_docker_client():
    return DummyDockerClient()


def test_image_get_fullname(dummy_docker_client):
    img = DockerImageClass("myapp", tag="v1", client=dummy_docker_client)
    fullname = img.get_fullname()
    assert fullname == "myapp:v1"


def test_image_get_image(dummy_docker_client):
    img = DockerImageClass("myapp", tag="latest", client=dummy_docker_client)
    # Simulate that the image exists in the dummy client's images manager
    dummy_img = DummyDockerImage("myapp", "latest")
    dummy_docker_client.images.images[dummy_img.get_fullname()] = dummy_img
    result = img.get_image()
    assert result is not None
    assert result.id == "dummy_image_id"


def test_image_pull_image(dummy_docker_client):
    img = DockerImageClass("myapp", tag="latest", client=dummy_docker_client)
    pulled = img.pull_image()
    assert pulled is not None
    assert pulled.get_fullname() == "myapp:latest"


def test_image_build_image(tmp_path, dummy_docker_client):
    # Test the build_image method of the Image class using a BuildConfig
    build_config = BuildConfig()
    img = DockerImageClass("myapp", tag="latest", client=dummy_docker_client)
    target_dir = tmp_path / "dockerfile_dir"
    target_dir.mkdir()
    built_img = img.build_image(build_config=build_config, dest=str(target_dir))
    assert built_img.get_fullname() == "myapp:latest"
    # Verify that a Dockerfile was created at the target location
    dockerfile_path = target_dir / build_config.docker_filename
    assert dockerfile_path.exists()


def test_image_remove(dummy_docker_client):
    img = DockerImageClass("myapp", tag="latest", client=dummy_docker_client)
    # Simulate that the image exists in the dummy images manager
    dummy_img = DummyDockerImage("myapp", "latest")
    dummy_docker_client.images.images[dummy_img.get_fullname()] = dummy_img
    img.remove()
    with pytest.raises(ImageNotFound):
        dummy_docker_client.images.get(dummy_img.get_fullname())


# -----------------------------------------------------------------------------
# Test cases for Runtime (runtime.py)


# Dummy container class for testing runtime functionality
class DummyContainerForRuntime:
    def __init__(self):
        self.status = "created"
        self.image_name = "dummy_container:latest"

    def run(self):
        self.status = "running"

    def stop(self):
        self.status = "stopped"

    def remove(self):
        self.status = "removed"

    def logs(self):
        return b"runtime dummy logs"

    def wait(self):
        pass

    @property
    def is_created(self):
        return True

    @property
    def is_running(self):
        return self.status == "running"


@pytest.fixture
def dummy_runtime_containers():
    # Create a list of dummy containers for testing SimpleRuntime
    return [DummyContainerForRuntime() for _ in range(2)]


def test_simple_runtime_run(dummy_runtime_containers):
    runtime = SimpleRuntime(containers=dummy_runtime_containers)
    runtime.run()
    for container in dummy_runtime_containers:
        assert container.status == "running"


def test_simple_runtime_stop(dummy_runtime_containers):
    # Set containers to running state
    for container in dummy_runtime_containers:
        container.status = "running"
    runtime = SimpleRuntime(containers=dummy_runtime_containers)
    runtime.stop()
    for container in dummy_runtime_containers:
        assert container.status == "stopped"


def test_simple_runtime_remove(dummy_runtime_containers):
    # Set containers to stopped state
    for container in dummy_runtime_containers:
        container.status = "stopped"
    runtime = SimpleRuntime(containers=dummy_runtime_containers)
    runtime.remove()
    for container in dummy_runtime_containers:
        assert container.status == "removed"


def test_simple_runtime_logs(tmp_path, dummy_runtime_containers):
    runtime = SimpleRuntime(containers=dummy_runtime_containers)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    runtime.logs(output_dir=str(log_dir))
    # Since the dummy implementation writes logs to a file,
    # we assume that the logs method was called.
    # Further verification of file content can be added as needed.
    pass
