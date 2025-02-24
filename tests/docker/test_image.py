import json
import docker
from unittest.mock import patch
from pathlib import Path
from ures.docker.image import ImageConstructor, ImageOrchestrator, Image


class TestImageConstructor:
    def test_home_dir_default(self, build_config):
        config = build_config(user=None, base_image="python:3.9")
        constructor = ImageConstructor(config)
        assert constructor.home_dir == Path("/root")

    def test_home_dir_with_user(self, build_config):
        config = build_config(user="alice", base_image="python:3.9")
        constructor = ImageConstructor(config)
        assert constructor.home_dir == Path("/home/alice")

    def test_minimal_dockerfile(self, build_config):
        config = build_config(base_image="python:3.9")
        constructor = ImageConstructor(config)
        expected = ["FROM python:3.9"]
        assert constructor.content == expected

    def test_dockerfile_with_platform(self, build_config):
        config = build_config(base_image="ubuntu:20.04", platform="linux/amd64")
        constructor = ImageConstructor(config)
        expected = "FROM --platform=linux/amd64 ubuntu:20.04"
        assert constructor.content[0] == expected

    def test_dockerfile_with_labels(self, build_config):
        labels = [("maintainer", "test@example.com"), ("version", "2.0")]
        config = build_config(base_image="python:3.9", labels=labels)
        constructor = ImageConstructor(config)
        expected = [
            "FROM python:3.9",
            'LABEL "maintainer"="test@example.com"',
            'LABEL "version"="2.0"',
        ]
        assert constructor.content[:3] == expected

    def test_dockerfile_with_user_and_uid(self, build_config):
        config = build_config(user="bob", uid=1000, base_image="python:3.9")
        constructor = ImageConstructor(config)
        expected = [
            "FROM python:3.9",
            "ARG HOME_DIR=/home/bob",
            "ARG USER_NAME=bob",
            "ARG UID=1000",
            "RUN useradd -m -u $UID -s /bin/bash -d $HOME_DIR $USER_NAME",
            "RUN chown -R $USER_NAME:$USER_NAME $HOME_DIR",
            "USER $USER_NAME",
            "WORKDIR $HOME_DIR",
        ]
        assert constructor.content == expected

    def test_dockerfile_with_user_no_uid(self, build_config):
        config = build_config(user="bob", base_image="python:3.9")
        constructor = ImageConstructor(config)
        expected = [
            "FROM python:3.9",
            "ARG HOME_DIR=/home/bob",
            "ARG USER_NAME=bob",
            "USER $USER_NAME",
            "WORKDIR $HOME_DIR",
        ]
        assert constructor.content == expected

    def test_dockerfile_with_system_dependencies_apt(self, build_config):
        config = build_config(
            base_image="python:3.9",
            sys_dependencies=["curl", "git"],
            sys_deps_manager="apt",
        )
        constructor = ImageConstructor(config)
        expected = [
            "FROM python:3.9",
            "RUN apt update && apt install -y curl git && apt-get clean && rm -rf /var/lib/apt/lists/*",
        ]
        assert constructor.content == expected

    def test_dockerfile_with_system_dependencies_other(self, build_config):
        config = build_config(
            base_image="python:3.9", sys_dependencies=["curl"], sys_deps_manager="yum"
        )
        constructor = ImageConstructor(config)
        expected = [
            "FROM python:3.9",
            "RUN yum update && yum install -y curl",
        ]
        assert constructor.content == expected

    def test_dockerfile_with_python_dependencies_pip(self, build_config):
        config = build_config(
            base_image="python:3.9",
            python_dependencies=["requests"],
            python_deps_manager="pip",
        )
        constructor = ImageConstructor(config)
        expected = [
            "FROM python:3.9",
            "RUN pip install --upgrade pip && pip install --no-cache-dir requests  && rm -rf /tmp/* /var/tmp/*",
        ]
        assert constructor.content == expected

    def test_dockerfile_with_python_dependencies_conda(self, build_config):
        config = build_config(
            base_image="python:3.9",
            python_dependencies=["requests"],
            python_deps_manager="conda",
        )
        constructor = ImageConstructor(config)
        # Since conda support is not implemented, no extra command is added.
        expected = ["FROM python:3.9"]
        assert constructor.content == expected

    def test_dockerfile_with_copies_relative(self, build_config):
        config = build_config(
            user="charlie",
            base_image="python:3.9",
            copies=[{"src": "app.py", "dest": "app/app.py"}],
        )
        constructor = ImageConstructor(config)
        expected = [
            "FROM python:3.9",
            "ARG HOME_DIR=/home/charlie",
            "ARG USER_NAME=charlie",
            "USER $USER_NAME",
            "WORKDIR $HOME_DIR",
            "COPY --chown=$USER_NAME app.py /home/charlie/app/app.py",
        ]
        assert constructor.content == expected

    def test_dockerfile_with_copies_absolute(self, build_config):
        config = build_config(
            user="charlie",
            base_image="python:3.9",
            copies=[{"src": "app.py", "dest": "/opt/app/app.py"}],
        )
        constructor = ImageConstructor(config)
        expected = [
            "FROM python:3.9",
            "ARG HOME_DIR=/home/charlie",
            "ARG USER_NAME=charlie",
            "USER $USER_NAME",
            "WORKDIR $HOME_DIR",
            "COPY --chown=$USER_NAME app.py /opt/app/app.py",
        ]
        assert constructor.content == expected

    def test_dockerfile_with_environment(self, build_config):
        config = build_config(
            base_image="python:3.9", environment={"DEBUG": "1", "ENV": "production"}
        )
        constructor = ImageConstructor(config)
        expected = [
            "FROM python:3.9",
            "ENV DEBUG=1",
            "ENV ENV=production",
        ]
        assert constructor.content == expected

    def test_dockerfile_with_entrypoint_and_cmd(self, build_config):
        config = build_config(
            base_image="python:3.9",
            entrypoint=["/entrypoint.sh"],
            cmd=["python", "app.py"],
        )
        constructor = ImageConstructor(config)
        expected_entrypoint = "ENTRYPOINT " + json.dumps(["/entrypoint.sh"])
        # Verify that the entrypoint is set and that CMD is omitted.
        assert expected_entrypoint in constructor.content
        assert not any(cmd.startswith("CMD") for cmd in constructor.content)

    def test_dockerfile_with_cmd_only(self, build_config):
        config = build_config(base_image="python:3.9", cmd=["python", "app.py"])
        constructor = ImageConstructor(config)
        expected_cmd = "CMD " + json.dumps(["python", "app.py"])
        assert expected_cmd in constructor.content

    def test_save(self, tmp_path, build_config):
        config = build_config(
            user="dave", base_image="python:3.9", docker_filename="Dockerfile.test"
        )
        constructor = ImageConstructor(config)
        file_path = constructor.save(tmp_path)
        expected_file = tmp_path / "Dockerfile.test"
        assert file_path == expected_file
        written_content = expected_file.read_text()
        assert written_content == "\n".join(constructor.content)
        print(file_path)


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
    def test_image_orchestrator_add_and_sort(self, docker_client , build_config_image):
        orch = ImageOrchestrator(client=docker_client)
        img1 = Image("baseapp", tag="v1", client=docker_client)
        img2 = Image("childapp", tag="v1", client=docker_client)
        # Add base image first
        orch.add_image(img1, build_config_image)
        # For child, set base image to img1
        orch.add_image(img2, build_config_image, base=img1)
        sorted_list = orch._topological_sort()
        # Base image should come before child image
        assert sorted_list.index(img1.get_fullname()) < sorted_list.index(img2.get_fullname())