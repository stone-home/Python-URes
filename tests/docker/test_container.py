import pytest
import docker
from ures.docker.container import Container
from ures.docker.conf import RuntimeConfig

# Define some Dummy objects to simulate docker interactions


class DummyContainer:
    def __init__(self, container_id="container_123"):
        self.id = container_id
        self.status = "created"  # Initial status

    def start(self):
        self.status = "running"

    def stop(self):
        self.status = "stopped"

    def remove(self):
        self.status = "removed"

    def logs(self):
        return b"dummy logs"

    def wait(self):
        pass


class DummyContainersManager:
    def __init__(self):
        self.created = None

    def create(self, **kwargs):
        container = DummyContainer()
        container.run_params = kwargs
        self.created = container
        return container

    def get(self, container_id):
        if self.created and self.created.id == container_id:
            return self.created
        raise docker.errors.NotFound("Container not found")


class DummyNetwork:
    def __init__(self, name):
        self.name = name
        # Set a fixed subnet configuration
        self.attrs = {"IPAM": {"Config": [{"Subnet": "172.17.0.0/16"}]}}
        self.connected = {}

    def connect(self, container, ipv4_address):
        self.connected[container.id] = ipv4_address


class DummyNetworksManager:
    def __init__(self):
        self.networks = {}

    def create(self, **kwargs):
        network = DummyNetwork(kwargs["name"])
        self.networks[kwargs["name"]] = network
        return network

    def get(self, name):
        if name in self.networks:
            return self.networks[name]
        raise docker.errors.NotFound("Network not found")


class DummyDockerClient:
    def __init__(self):
        self.containers = DummyContainersManager()
        self.networks = DummyNetworksManager()


# Define a simple Dummy Image for testing Container's image-related properties
class DummyImage:
    def __init__(self, image_name="test_image", tag="latest"):
        self._image_name = image_name
        self.tag = tag

    def get_fullname(self, tag=None):
        if tag is None:
            tag = self.tag
        return f"{self._image_name}:{tag}"


# ------------------------- Test Container Properties -------------------------
class TestContainerProperties:
    @pytest.fixture
    def dummy_client(self):
        return DummyDockerClient()

    @pytest.fixture
    def dummy_image(self):
        return DummyImage("test_image", "latest")

    @pytest.fixture
    def container_instance(self, dummy_image, dummy_client):
        return Container(dummy_image, client=dummy_client)

    def test_image_name_property(self, container_instance, dummy_image):
        # The image_name property should return the result of image.get_fullname()
        assert container_instance.image_name == dummy_image.get_fullname()

    def test_is_created_initial(self, container_instance):
        # Initially, _container is None, so is_created should be False
        assert container_instance.is_created is False

    def test_status_when_not_created(self, container_instance, dummy_client):
        # When the container does not exist, status should return "removed"
        # Simulate a _container that does not exist in DummyContainersManager
        container_instance._container = DummyContainer("non_exist")
        # Even though DummyContainersManager.get raises an exception,
        # the Container.status method catches the exception and returns "removed"
        assert container_instance.status == "removed"


# ------------------------- Test Container Lifecycle Methods -------------------------
class TestContainerLifecycle:
    @pytest.fixture
    def dummy_client(self):
        return DummyDockerClient()

    @pytest.fixture
    def dummy_image(self):
        return DummyImage("test_image", "latest")

    @pytest.fixture
    def runtime_config(self):
        # Create a minimal RuntimeConfig object with network-related required fields
        return RuntimeConfig(
            image_name="test_image:latest",
            subnet="test_network",
            subnet_mask="172.17.0.0/16",
            subnet_gateway="172.17.0.1",
            ipv4="172.17.0.2",
            network_mode="bridge",
            detach=True,
            remove=False,
        )

    @pytest.fixture
    def container_instance(self, dummy_image, dummy_client):
        return Container(dummy_image, client=dummy_client)

    def test_create_sets_container(
        self, container_instance, runtime_config, dummy_client
    ):
        # After calling create, container_instance._container should not be None,
        # and the run parameters (including image_name) should be updated if necessary.
        container_instance.create(runtime_config)
        assert container_instance._container is not None
        # Check that the container returned by DummyContainersManager.create is used
        created = dummy_client.containers.created
        assert created is container_instance._container
        # Check that the network connection is established (called in _connect_to_network)
        network = dummy_client.networks.get(runtime_config.subnet)
        assert container_instance._container.id in network.connected
        # And the connected IP address matches the configuration
        assert (
            network.connected[container_instance._container.id] == runtime_config.ipv4
        )

    def test_run_without_creation_raises(self, container_instance):
        # If _container is None, then calling run() should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            container_instance.run()
        assert "Container has not been created" in str(exc_info.value)

    def test_run_already_running_raises(self, container_instance, dummy_client):
        # Simulate a container that has been created and is already running
        dummy_container = DummyContainer()
        dummy_container.status = "running"
        container_instance._container = dummy_container
        # Register dummy_container in DummyContainersManager so that status is correctly retrieved
        dummy_client.containers.created = dummy_container
        with pytest.raises(RuntimeError) as exc_info:
            container_instance.run()
        assert "Container already running" in str(exc_info.value)

    def test_run_starts_container(self, container_instance, dummy_client):
        # Simulate a container that is created but not started. After calling run, status should be "running".
        dummy_container = DummyContainer()
        dummy_container.status = "created"
        container_instance._container = dummy_container
        container_instance.run()
        assert container_instance._container.status == "running"

    def test_stop_remove_logs_wait(self, container_instance, dummy_client):
        # First set _container to a DummyContainer
        dummy_container = DummyContainer()
        container_instance._container = dummy_container

        # Test the stop() method
        container_instance.stop()
        assert dummy_container.status == "stopped"

        # Test the logs() method
        logs = container_instance.logs()
        assert logs == b"dummy logs"

        # Test the wait() method (no return value; just ensure no exception is thrown)
        container_instance.wait()

        # Test the remove() method
        container_instance.remove()
        # After remove() is called, _container should be None
        assert container_instance._container is None


# ------------------------- Test Container Network-Related Logic -------------------------
class TestContainerNetwork:
    @pytest.fixture
    def dummy_client(self):
        return DummyDockerClient()

    @pytest.fixture
    def dummy_image(self):
        return DummyImage("test_image", "latest")

    @pytest.fixture
    def runtime_config(self):
        # Configuration with network-related parameters specified
        return RuntimeConfig(
            image_name="test_image:latest",
            subnet="test_network",
            subnet_mask="172.17.0.0/16",
            subnet_gateway="172.17.0.1",
            ipv4="172.17.0.3",
            network_mode="bridge",
            detach=True,
            remove=False,
        )

    @pytest.fixture
    def container_instance(self, dummy_image, dummy_client):
        return Container(dummy_image, client=dummy_client)

    def test_create_subnet_creates_network(
        self, container_instance, runtime_config, dummy_client
    ):
        # When create() is called, if the specified network does not exist,
        # _create_subnet should create the network.
        # Ensure the network returned by _create_subnet has the correct subnet configuration.
        network = container_instance._create_subnet(runtime_config)
        assert (
            network.attrs["IPAM"]["Config"][0]["Subnet"] == runtime_config.subnet_mask
        )

    def test_connect_to_network_uses_existing_network(
        self, container_instance, runtime_config, dummy_client
    ):
        # Manually create a network and add it to dummy_client.networks
        network = dummy_client.networks.create(
            name=runtime_config.subnet, driver=runtime_config.network_mode, ipam=None
        )
        dummy_container = DummyContainer()
        # After calling _connect_to_network, the container should be connected to the existing network
        container_instance._connect_to_network(dummy_container, runtime_config)
        # Check that the network's connected dictionary contains the container with the specified ipv4
        assert dummy_container.id in network.connected
        assert network.connected[dummy_container.id] == runtime_config.ipv4
