import json
import docker
import pytest
from unittest.mock import patch
from pathlib import Path
from ures.docker.image import ImageConstructor, ImageOrchestrator, Image
from ures.docker.conf import BuildConfig


@pytest.fixture
def sample_build_config():
    """
    Provides a sample BuildConfig for testing.

    Returns:
        BuildConfig: A sample build configuration with test values.

    Example:
        >>> config = sample_build_config()
        >>> config.user
        'testuser'
    """
    return BuildConfig(
        base_image="python:3.10-slim",
        user="testuser",
        sys_dependencies=["curl"],
        sys_deps_manager="apt",
        python_dependencies=["flask"],
        python_deps_manager="pip",
        labels=[("version", "1.0")],
        docker_filename="Dockerfile",
    )


class TestImageConstructor:
    def test_home_dir(self, sample_build_config):
        """
        Test that the home_dir property returns the correct path based on the user.

        Example:
            >>> constructor.home_dir == PosixPath("/home/testuser")
            True
        """
        constructor = ImageConstructor(sample_build_config)
        expected = Path(f"/home/{sample_build_config.user}")
        assert constructor.home_dir == expected

    def test_dockerfile_content_generated(self, sample_build_config):
        """
        Test that the Dockerfile content is generated and contains expected commands.

        Example:
            >>> any("FROM" in line for line in constructor.content)
            True
        """
        constructor = ImageConstructor(sample_build_config)
        content = constructor.content
        # Verify that a FROM command is present.
        assert any(
            line.startswith("FROM") for line in content
        ), "Dockerfile should have a FROM command"
        # Verify that the user and workdir commands are present.
        assert any("USER" in line for line in content), "Dockerfile should set a USER"
        assert any(
            "WORKDIR" in line for line in content
        ), "Dockerfile should set a WORKDIR"

    def test_save_dockerfile(self, tmp_path, sample_build_config):
        """
        Test that the Dockerfile is saved correctly to the specified destination.

        Args:
            tmp_path (Path): A temporary directory provided by pytest.

        Example:
            >>> saved_file.exists()
            True
        """
        constructor = ImageConstructor(sample_build_config)
        saved_path = constructor.save(tmp_path)
        # Verify that the file exists.
        assert saved_path.exists(), "Saved Dockerfile should exist"
        # Read file content and verify non-empty and expected commands.
        content = saved_path.read_text()
        assert len(content) > 0, "Dockerfile content should not be empty"
        assert "FROM" in content, "Dockerfile should contain a FROM command"
        assert (
            "ARG HOME_DIR" in content
        ), "Dockerfile should contain the HOME_DIR argument"

    def test_add_command(self, sample_build_config):
        """
        Test that _add_command appends a command to the Dockerfile content.

        Example:
            >>> constructor.content[-1] == "RUN echo Test"
            True
        """
        constructor = ImageConstructor(sample_build_config)
        initial_length = len(constructor.content)
        constructor._add_command("RUN echo Test")
        # Ensure the command is appended at the end.
        assert len(constructor.content) == initial_length + 1
        assert constructor.content[-1] == "RUN echo Test"


class TestImage:
    """Test cases for the Image class."""

    def test_image_initialization(self, test_image):
        """Test that the Image object is initialized correctly."""
        assert test_image.name == "test_image"
        assert test_image.tag == "latest"

    def test_get_fullname(self, test_image):
        """Test getting the full image name with a tag."""
        assert test_image.get_fullname() == "test_image:latest"
        assert test_image.get_fullname(tag="1.0") == "test_image:1.0"

    def test_image_existence(self, test_image, docker_client, mock_image):
        """Test checking if an image exists."""
        docker_client.images.get.return_value = mock_image
        assert test_image.exist is True

    def test_image_not_found(self, test_image, docker_client):
        """Test checking if an image does not exist."""
        docker_client.images.get.side_effect = docker.errors.ImageNotFound("Not found")
        assert test_image.exist is False

    def test_get_image(self, test_image, docker_client, mock_image):
        """Test retrieving an image."""
        docker_client.images.get.return_value = mock_image
        image = test_image.get_image()
        assert image is mock_image

    def test_get_image_not_found(self, test_image, docker_client):
        """Test retrieving a non-existent image."""
        docker_client.images.get.side_effect = docker.errors.ImageNotFound("Not found")
        assert test_image.get_image() is None

    def test_pull_image(self, test_image, docker_client, mock_image):
        """Test pulling an image."""
        docker_client.images.pull.return_value = mock_image
        pulled_image = test_image.pull_image()
        assert pulled_image is mock_image
        docker_client.images.pull.assert_called_with("test_image", tag="latest")

    def test_pull_image_fail(self, test_image, docker_client):
        """Test pulling an image with failure."""
        docker_client.images.pull.side_effect = docker.errors.APIError("Pull failed")
        assert test_image.pull_image() is None

    def test_image_id_property(self, test_image):
        """Test retrieving the image ID."""
        assert test_image.id == "mock_id"

    def test_image_architecture_property(self, test_image):
        """Test retrieving the image architecture."""
        assert test_image.architecture == "amd64"

    def test_image_size_property(self, test_image):
        """Test retrieving the image size."""
        assert test_image.image_size == 12345678

    def test_image_labels_property(self, test_image):
        """Test retrieving the image labels."""
        assert test_image.labels == {"maintainer": "test@example.com"}

    def test_build_image(
        self, test_image, build_config_image, tmp_path, docker_client, mock_image
    ):
        """Test building an image."""
        docker_client.images.build.return_value = (mock_image, [])
        build_context = tmp_path / "context"
        build_context.mkdir()
        built_image = test_image.build_image(
            build_config=build_config_image, dest=tmp_path, build_context=build_context
        )
        assert built_image is mock_image
        docker_client.images.build.assert_called()

    def test_remove_image(self, test_image, docker_client):
        """Test removing an image."""
        test_image.remove()
        docker_client.images.remove.assert_called_with(
            image="test_image:latest", force=False, noprune=False
        )

    def test_remove_image_force(self, test_image, docker_client):
        """Test force removing an image."""
        test_image.remove(force=True)
        docker_client.images.remove.assert_called_with(
            image="test_image:latest", force=True, noprune=False
        )

    def test_info_method(self, test_image):
        """Test displaying image info."""
        with patch("builtins.print") as mock_print:
            test_image.info()
            assert mock_print.called


class TestImageOrchestrator:
    def test_image_orchestrator_add_and_sort(self, docker_client, build_config_image):
        orch = ImageOrchestrator(client=docker_client)
        img1 = Image("baseapp", tag="v1", client=docker_client)
        img2 = Image("childapp", tag="v1", client=docker_client)
        # Add base image first
        orch.add_image(img1, build_config_image)
        # For child, set base image to img1
        orch.add_image(img2, build_config_image, base=img1)
        sorted_list = orch._topological_sort()
        # Base image should come before child image
        assert sorted_list.index(img1.get_fullname()) < sorted_list.index(
            img2.get_fullname()
        )
