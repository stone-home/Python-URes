import logging
import json
import docker
import docker.errors
from pathlib import Path
from typing import Optional, Union, List
from docker.models.images import Image as DockerImage
from ures.tools.decorator import check_instance_variable
from ures.string import format_memory
from .conf import BuildConfig

logger = logging.getLogger(__name__)


class ImageConstructor:
    def __init__(self, config: BuildConfig):
        self._config = config
        self._dockerfile_content = []
        self._build_dockerfile()

    @property
    def home_dir(self) -> Path:
        """Returns the home directory path based on the configured user."""
        home_dir = (
            f"/home/{self._config.user}" if self._config.user else "/root"
        )  # Default to /root if no user
        return Path(home_dir)

    @property
    def content(self) -> List[str]:
        return self._dockerfile_content

    def _add_command(self, command: str):
        """Appends a command to the Dockerfile content."""
        logger.debug(f"Adding command: {command}")
        self._dockerfile_content.append(command)

    def save(self, dest: Union[str, Path]) -> Path:
        """Saves the generated Dockerfile to the specified destination.

        If the destination is a directory, the Dockerfile will be named "Dockerfile"
        and placed inside that directory. Creates parent directories if needed.
        """
        dest_path = Path(dest) if isinstance(dest, str) else dest
        if dest_path.is_dir():
            dest_path = dest_path / self._config.docker_filename

        logger.info(f"Saving Dockerfile to: {dest_path}")
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        with open(dest_path, "w") as f:
            f.write("\n".join(self._dockerfile_content))
        return dest_path

    def _set_base_image(self):
        """Sets the base image in the Dockerfile, including platform if specified."""
        base_image = self._config.base_image
        if self._config.platform:
            base_image = f"--platform={self._config.platform} {base_image}"
        self._add_command(f"FROM {base_image}")

    def _set_labels(self):
        """Sets labels in the Dockerfile."""
        if self._config.labels:
            for key, value in self._config.labels:
                self._add_command(f'LABEL "{key}"="{value}"')

    def _set_user_and_workdir(self):
        """Sets the user and working directory in the Dockerfile."""
        user = self._config.user
        uid = self._config.uid

        if user:
            self._add_command(f"ARG HOME_DIR={self.home_dir}")  # Define HOME_DIR early
            self._add_command(f"ARG USER_NAME={user}")

            if uid:
                self._add_command(f"ARG UID={uid}")
                self._add_command(
                    f"RUN useradd -m -u $UID -s /bin/bash -d $HOME_DIR $USER_NAME"
                )
                self._add_command(f"RUN chown -R $USER_NAME:$USER_NAME $HOME_DIR")

            self._add_command(f"USER $USER_NAME")
            self._add_command(f"WORKDIR $HOME_DIR")  # Set workdir *after* setting user

    def _set_system_dependencies(self):
        """Installs system dependencies."""
        sys_deps = self._config.sys_dependencies
        manager = self._config.sys_deps_manager

        if sys_deps:
            deps_string = " ".join(sys_deps)
            command = f"RUN {manager} update && {manager} install -y {deps_string}"
            if manager == "apt":  # Add cleanup for apt
                command += " && apt-get clean && rm -rf /var/lib/apt/lists/*"
            self._add_command(command)

    def _set_python_dependencies(self):
        """Installs Python dependencies."""
        python_deps = self._config.python_dependencies
        manager = self._config.python_deps_manager

        if python_deps:
            deps_string = " ".join(python_deps)
            if manager == "pip":
                command = (
                    f"RUN pip install --upgrade pip && pip install --no-cache-dir {deps_string} "
                    f" && rm -rf /tmp/* /var/tmp/*"
                )
                self._add_command(command)
            elif manager == "conda":  # Add conda support if needed
                # todo
                pass
            else:
                logger.warning(f"Unsupported python package manager: {manager}")

    def _set_copies(self):
        """Copies files into the image."""
        if self._config.copies:
            for copy_spec in self._config.copies:
                src = Path(copy_spec["src"])
                dest = Path(copy_spec["dest"])

                # Handle relative destination paths
                if dest.is_absolute() is False:
                    dest = self.home_dir.joinpath(dest)

                self._add_command(f"COPY --chown=$USER_NAME {src} {dest}")

    def _set_environment(self):
        """Sets environment variables."""
        if self._config.environment:
            for key, value in self._config.environment.items():
                self._add_command(f"ENV {key}={value}")

    def _set_entrypoint(self):
        """Sets the entrypoint."""
        if self._config.entrypoint:
            self._add_command(f"ENTRYPOINT {json.dumps(self._config.entrypoint)}")

    def _set_cmd(self):
        """Sets the command (only if entrypoint is not set)."""
        if self._config.cmd and not self._config.entrypoint:
            self._add_command(f"CMD {json.dumps(self._config.cmd)}")

    def _build_dockerfile(self):
        """Generates the Dockerfile content."""
        self._set_base_image()
        self._set_labels()
        self._set_user_and_workdir()  # combined user and workdir
        self._set_system_dependencies()
        self._set_python_dependencies()
        self._set_copies()
        self._set_environment()
        self._set_entrypoint()
        self._set_cmd()


class Image:
    """Represents a Docker image, providing methods to manage and manipulate images.

    Attributes:
        _image_name (str): The name of the Docker image.
        _tag (str): The tag of the Docker image, default is "latest".
        _client (docker.DockerClient): The Docker client instance.
        _image (Optional[DockerImage]): The Docker image object.
    """

    def __init__(
        self,
        image_name: str,
        tag: Optional[str] = None,
        client: docker.DockerClient = None,
    ):
        """Initializes an Image instance.

        Args:
            image_name (str): The name of the Docker image.
            tag (Optional[str], optional): The image tag. Defaults to "latest".
        """
        self._image_name = image_name
        self._tag = tag or "latest"
        self._client = client or docker.from_env()
        self._image: Optional[DockerImage] = None

    @property
    def name(self) -> str:
        """Gets the name of the image.

        Returns:
            str: The name of the image.
        """
        return self._image_name

    @property
    def tag(self) -> str:
        """Gets the tag of the image.

        Returns:
            str: The tag of the image.
        """
        return self._tag

    @property
    def exist(self) -> bool:
        """Checks whether the image exists locally.

        Returns:
            bool: True if the image exists, False otherwise.
        """
        return self.get_image() is not None

    @property
    def image(self) -> Optional[DockerImage]:
        """Gets the Docker image object if it exists.

        Returns:
            Optional[DockerImage]: The Docker image object or None if not found.
        """
        return self._image

    @property
    @check_instance_variable("image")
    def id(self) -> str:
        """Gets the ID of the image.

        Returns:
            str: The image ID.
        """
        return self.image.id

    @property
    @check_instance_variable("image")
    def architecture(self) -> str:
        """Gets the architecture of the image.

        Returns:
            str: The image architecture.
        """
        return self.image.attrs["Architecture"]

    @property
    @check_instance_variable("image")
    def image_size(self) -> int:
        """Gets the size of the image in MB.

        Returns:
            int: The image size in MB.
        """
        return int(self.image.attrs["Size"])

    @property
    @check_instance_variable("image")
    def labels(self) -> dict:
        """Gets the labels associated with the image.

        Returns:
            dict: A dictionary containing image labels.
        """
        return self.image.labels

    def get_fullname(self, tag: Optional[str] = None) -> str:
        """Gets the full name of the image including the tag.

        Args:
            tag (Optional[str], optional): The image tag. Defaults to the instance's tag.

        Returns:
            str: The full image name with the tag.
        """
        if tag is None:
            tag = self._tag
        return f"{self._image_name}:{tag}"

    def get_image(self, tag: Optional[str] = None) -> Optional[DockerImage]:
        """Retrieves the Docker image from the local repository.

        Args:
            tag (Optional[str], optional): The tag of the image. Defaults to the instance's tag.

        Returns:
            Optional[DockerImage]: The Docker image object if found, otherwise None.
        """
        image_name = self.get_fullname(tag=tag)
        logger.info(f"Getting image {image_name}")
        try:
            image = self._client.images.get(image_name)
            if tag is None or tag == self._tag:
                self._image = image
            return image
        except docker.errors.ImageNotFound:
            return None
        except docker.errors.APIError as e:
            print(f"Error accessing Docker API: {e}")
            return None

    def pull_image(self, tag: Optional[str] = None) -> Optional[DockerImage]:
        """Pulls the Docker image from a remote repository.

        Args:
            tag (Optional[str], optional): The image tag to pull. Defaults to the instance's tag.

        Returns:
            Optional[DockerImage]: The pulled Docker image object if successful, otherwise None.
        """
        tag = tag or self._tag
        logger.info(f"Pulling image {self.get_fullname(tag=tag)}")
        try:
            image = self._client.images.pull(self._image_name, tag=tag)
        except docker.errors.APIError as e:
            logger.error(f"Error for pulling image {self.get_fullname(tag=tag)}")
        else:
            self._image = image
            return image

    def build_image(
        self,
        build_context: Union[str, Path],
        build_config: BuildConfig,
        dest: Union[str, Path],
    ) -> DockerImage:
        """Builds a Docker image from a specified build context.

        Args:
            build_context (Union[str, Path]): The path to the build context.
            build_config (BuildConfig): The build configuration settings.
            dest (Union[str, Path]): The destination where the Dockerfile is saved.

        Returns:
            DockerImage: The built Docker image.
        """
        build_context = Path(build_context)
        dest = Path(dest)
        builder = ImageConstructor(build_config)
        docker_path = builder.save(dest)
        image_name = self.get_fullname()

        args = {
            "path": str(build_context),
            "tag": image_name,
            "dockerfile": str(docker_path),
            "nocache": True,
        }

        logger.info(
            f"Building image {image_name} with dockerfile {docker_path} in context {build_context}"
        )
        try:
            image, build_log = self._client.images.build(**args)
        except docker.errors.BuildError as e:
            logger.error(f"Failed to build image {image_name}")
            for log in e.build_log:
                logger.error(log)
            raise e
        else:
            logger.info(f"Image {image_name} is built successfully!")
            self._image = image

        for line in build_log:
            logger.debug(line)
        return image

    def remove(
        self, tag: Optional[str] = None, force: bool = False, noprune: bool = False
    ):
        """Removes the specified Docker image.

        Args:
            tag (Optional[str], optional): The image tag to remove. Defaults to the instance's tag.
            force (bool, optional): Whether to force remove the image. Defaults to False.
        """
        image_name = self.get_fullname(tag=tag)
        args = {
            "image": image_name,
            "force": force,
            "noprune": noprune,
        }
        try:
            self._client.images.remove(**args)
        except docker.errors.APIError as e:
            logger.error(f"Failed to remove image {image_name}. Msg: {e}")
        finally:
            if self.exist is True:
                logger.error(f"Removing image {image_name} failed")
            else:
                logger.info(f"Removing image {image_name} succeeded")

    def info(self):
        """Prints detailed information about the image."""
        print(
            f"\033[1;33m====================================== Image Info ===============================================\033[0m"
        )
        print(f"Name: {self.name}")
        print(f"Image ID: {self.id}")
        print(f"Architecture: {self.architecture}")
        print(f"Image Size: {format_memory(self.image_size)}")
        print(f"Labels: {self.labels}")
