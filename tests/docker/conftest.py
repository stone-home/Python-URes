import pytest
import docker
from docker.models.images import Image as DockerImage
from unittest.mock import MagicMock, patch
from ures.docker.image import Image
from ures.docker.conf import BuildConfig


# Pytest fixture that returns a factory function for BuildConfig
@pytest.fixture
def build_config():
    def _build_config(**kwargs):
        return BuildConfig(**kwargs)

    return _build_config


@pytest.fixture
def docker_client():
    """Fixture to create a mock Docker client."""
    client = MagicMock(spec=docker.DockerClient)
    return client


@pytest.fixture
def mock_image():
    """Fixture to create a mock Docker image object."""
    image = MagicMock(spec=DockerImage)
    image.id = "mock_id"
    image.attrs = {"Architecture": "amd64", "Size": 12345678}
    image.labels = {"maintainer": "test@example.com"}
    return image


@pytest.fixture
def build_config_image():
    """Fixture to create a default BuildConfig object."""
    return BuildConfig(
        base_image="python:3.9",
        python_dependencies=["requests"],
        sys_dependencies=["curl"],
        cmd=["python", "app.py"],
    )


@pytest.fixture
def test_image(docker_client, mock_image):
    """Fixture to create an Image instance with a mocked Docker client."""
    with patch("docker.from_env", return_value=docker_client):
        image = Image(image_name="test_image", tag="latest")
        image._client = docker_client
        image._image = mock_image  # Assign the mock image
        return image
