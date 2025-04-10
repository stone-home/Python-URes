import logging
import json
import docker
import docker.errors
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Union, List, Dict
from docker.models.images import Image as DockerImage
from ures.tools.decorator import check_instance_variable
from ures.string import format_memory, unique_id
from ures.files import get_temp_dir_with_specific_path
from .conf import BuildConfig

logger = logging.getLogger(__name__)


class ImageConstructor:
    """
    Constructs a Dockerfile based on a provided build configuration.

    This class generates Dockerfile content by appending commands derived from the build
    configuration and provides methods to save the generated Dockerfile.
    """

    def __init__(self, config: BuildConfig):
        """
        Initializes the ImageConstructor with the given build configuration.

        Args:
            config (BuildConfig): The build configuration settings.

        Returns:
            None

        Example:
            >>> from ures.docker.conf import BuildConfig
            >>> config = BuildConfig(base_image="python:3.10-slim", user="appuser")
            >>> constructor = ImageConstructor(config)
        """
        self._config = config
        self._dockerfile_content: List[str] = []
        self._build_dockerfile()

    @property
    def home_dir(self) -> Path:
        """
        Returns the home directory path based on the configured user.

        Returns:
            Path: The home directory path (e.g. /home/{user} if user is specified, else /root).

        Example:
            >>> from ures.docker.conf import BuildConfig
            >>> config = BuildConfig(user="appuser")
            >>> constructor = ImageConstructor(config)
            >>> constructor.home_dir
            PosixPath('/home/appuser')
        """
        home_dir = f"/home/{self._config.user}" if self._config.user else "/root"
        return Path(home_dir)

    @property
    def content(self) -> List[str]:
        """
        Retrieves the generated Dockerfile content as a list of command strings.

        Returns:
            List[str]: The Dockerfile content lines.

        Example:
            >>> constructor = ImageConstructor(BuildConfig())
            >>> constructor.content  # Might include commands like 'FROM python:3.10-slim'
        """
        return self._dockerfile_content

    def _add_command(self, command: str):
        """
        Appends a command line to the Dockerfile content.

        Args:
            command (str): The Dockerfile command to add.

        Returns:
            None

        Example:
            >>> constructor = ImageConstructor(BuildConfig())
            >>> constructor._add_command("RUN echo Hello")
        """
        logger.debug(f"Adding command: {command}")
        self._dockerfile_content.append(command)

    def save(self, dest: Union[str, Path]) -> Path:
        """
        Saves the generated Dockerfile to the specified destination.

        If the destination is a directory, the Dockerfile will be named using the configuration's
        docker_filename and placed inside that directory.

        Args:
            dest (Union[str, Path]): The destination file path or directory.

        Returns:
            Path: The full path where the Dockerfile was saved.

        Example:
            >>> constructor = ImageConstructor(BuildConfig(docker_filename="Dockerfile"))
            >>> saved_path = constructor.save("/tmp")
            >>> saved_path.name  # Should be 'Dockerfile'
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
        """
        Sets the base image in the Dockerfile.

        If a platform is specified in the configuration, it is included in the FROM command.

        Returns:
            None

        Example:
            >>> constructor = ImageConstructor(BuildConfig(base_image="python:3.10-slim"))
            >>> constructor._set_base_image()
        """
        base_image = self._config.base_image
        if self._config.platform:
            base_image = f"--platform={self._config.platform} {base_image}"
        self._add_command(f"FROM {base_image}")

    def _set_labels(self):
        """
        Adds LABEL commands to the Dockerfile for each label defined in the build configuration.

        Returns:
            None

        Example:
            >>> config = BuildConfig(labels=[("version", "1.0")])
            >>> constructor = ImageConstructor(config)
            >>> constructor._set_labels()
        """
        if self._config.labels:
            for key, value in self._config.labels:
                self._add_command(f'LABEL "{key}"="{value}"')

    def _set_user_and_workdir(self):
        """
        Sets the user and working directory in the Dockerfile.

        This command sets arguments for HOME_DIR and USER_NAME, creates the user if UID is specified,
        and then sets the USER and WORKDIR.

        Returns:
            None

        Example:
            >>> config = BuildConfig(user="appuser", uid=1001)
            >>> constructor = ImageConstructor(config)
            >>> constructor._set_user_and_workdir()
        """
        user = self._config.user
        uid = self._config.uid
        self._add_command(f"ARG HOME_DIR={self.home_dir}")
        if user:
            self._add_command(f"ARG USER_NAME={user}")
            if uid:
                self._add_command(f"ARG UID={uid}")
                self._add_command(
                    "RUN id -u $USER_NAME >/dev/null 2>&1 || useradd -m -u $UID -s /bin/bash -d $HOME_DIR $USER_NAME"
                )
                self._add_command(f"RUN chown -R $USER_NAME:$USER_NAME $HOME_DIR")
            self._add_command(f"USER $USER_NAME")
        self._add_command(f"WORKDIR $HOME_DIR")

    def _set_system_dependencies(self):
        """
        Installs system dependencies in the Docker image using the specified package manager.

        Returns:
            None

        Example:
            >>> config = BuildConfig(sys_dependencies=["curl"], sys_deps_manager="apt")
            >>> constructor = ImageConstructor(config)
            >>> constructor._set_system_dependencies()
        """
        sys_deps = self._config.sys_dependencies
        manager = self._config.sys_deps_manager
        if sys_deps:
            deps_string = " ".join(sys_deps)
            command = f"RUN {manager} update && {manager} install -y {deps_string}"
            if manager == "apt":
                command += " && apt-get clean && rm -rf /var/lib/apt/lists/*"
            self._add_command(command)

    def _set_python_dependencies(self):
        """
        Installs Python dependencies in the Docker image using the specified Python package manager.

        Returns:
            None

        Example:
            >>> config = BuildConfig(python_dependencies=["flask"], python_deps_manager="pip")
            >>> constructor = ImageConstructor(config)
            >>> constructor._set_python_dependencies()
        """
        python_deps = self._config.python_dependencies
        manager = self._config.python_deps_manager
        if python_deps:
            deps_string = " ".join(python_deps)
            if manager == "pip":
                command = (
                    f"RUN pip install --upgrade pip && pip install --no-cache-dir {deps_string} "
                    f"&& rm -rf /tmp/* /var/tmp/*"
                )
                self._add_command(command)
            elif manager == "conda":
                # Placeholder for conda support.
                pass
            else:
                logger.warning(f"Unsupported python package manager: {manager}")

    def _set_run_commands(self):
        """
        Adds RUN commands to the Dockerfile for each command in the build configuration.

        Returns:
            None

        Example:
            >>> config = BuildConfig(run_commands=["echo Hello"])
            >>> constructor = ImageConstructor(config)
            >>> constructor._set_run_commands()
        """
        if self._config.run_commands:
            for command in self._config.run_commands:
                self._add_command(f"RUN {command}")

    def _set_copies(self):
        """
        Adds COPY commands to the Dockerfile for each file copy instruction in the build configuration.

        Returns:
            None

        Example:
            >>> config = BuildConfig(copies=[{"src": "app.py", "dest": "/app/app.py"}])
            >>> constructor = ImageConstructor(config)
            >>> constructor._set_copies()
        """
        if self._config.copies:
            for copy_spec in self._config.copies:
                src = Path(copy_spec["src"])
                dest = Path(copy_spec["dest"])
                if not dest.is_absolute():
                    dest = self.home_dir.joinpath(dest)
                if self._config.user is None:
                    self._add_command(f"COPY {src} {dest}")
                else:
                    self._add_command(f"COPY --chown=$USER_NAME {src} {dest}")

    def _set_environment(self):
        """
        Sets environment variables in the Dockerfile using the ENV command.

        Returns:
            None

        Example:
            >>> config = BuildConfig(environment={"DEBUG": "true"})
            >>> constructor = ImageConstructor(config)
            >>> constructor._set_environment()
        """
        if self._config.environment:
            for key, value in self._config.environment.items():
                self._add_command(f"ENV {key}={value}")

    def _set_entrypoint(self):
        """
        Sets the ENTRYPOINT in the Dockerfile using the build configuration.

        Returns:
            None

        Example:
            >>> config = BuildConfig(entrypoint=["python", "app.py"])
            >>> constructor = ImageConstructor(config)
            >>> constructor._set_entrypoint()
        """
        if self._config.entrypoint:
            self._add_command(f"ENTRYPOINT {json.dumps(self._config.entrypoint)}")

    def _set_cmd(self):
        """
        Sets the CMD in the Dockerfile if no ENTRYPOINT is specified.

        Returns:
            None

        Example:
            >>> config = BuildConfig(cmd=["python", "-m", "app"])
            >>> constructor = ImageConstructor(config)
            >>> constructor._set_cmd()
        """
        if self._config.cmd and not self._config.entrypoint:
            self._add_command(f"CMD {json.dumps(self._config.cmd)}")

    def _build_dockerfile(self):
        """
        Builds the complete Dockerfile content based on the build configuration.

        Returns:
            None

        Example:
            >>> constructor = ImageConstructor(BuildConfig())
            >>> constructor.content  # Contains all Dockerfile commands
        """
        self._set_base_image()
        self._set_labels()
        self._set_system_dependencies()
        self._set_python_dependencies()
        self._set_run_commands()
        self._set_user_and_workdir()
        self._set_copies()
        self._set_environment()
        self._set_entrypoint()
        self._set_cmd()


class Image:
    """
    Represents and manages a Docker image.

    Attributes:
        _image_name (str): The name of the Docker image.
        _tag (str): The tag of the Docker image (default is "latest").
        _client (docker.DockerClient): The Docker client instance.
        _image (Optional[DockerImage]): The Docker image object if available.
    """

    def __init__(
        self,
        image_name: str,
        tag: Optional[str] = None,
        client: docker.DockerClient = None,
    ):
        """
        Initializes an Image instance.

        Args:
            image_name (str): The name of the Docker image.
            tag (Optional[str], optional): The image tag. Defaults to "latest" if not provided.
            client (Optional[docker.DockerClient], optional): The Docker client to use. Defaults to docker.from_env().

        Example:
            >>> img = Image("myapp", tag="v1")
        """
        self._image_name = image_name
        self._tag = tag or "latest"
        self._client = client or docker.from_env()
        self._image: Optional[DockerImage] = None

    @property
    def name(self) -> str:
        """
        Gets the name of the image.

        Returns:
            str: The image name.

        Example:
            >>> img = Image("myapp")
            >>> img.name
            'myapp'
        """
        return self._image_name

    @property
    def tag(self) -> str:
        """
        Gets the tag of the image.

        Returns:
            str: The image tag.

        Example:
            >>> img = Image("myapp", tag="v1")
            >>> img.tag
            'v1'
        """
        return self._tag

    @property
    def exist(self) -> bool:
        """
        Checks if the image exists locally.

        Returns:
            bool: True if the image exists, False otherwise.

        Example:
            >>> img = Image("myapp")
            >>> img.exist  # Depends on local Docker images
        """
        return self.get_image() is not None

    @property
    def image(self) -> Optional[DockerImage]:
        """
        Gets the Docker image object.

        Returns:
            Optional[DockerImage]: The Docker image object if found, otherwise None.

        Example:
            >>> img = Image("myapp")
            >>> img.image  # Might return a DockerImage object if available
        """
        return self._image

    @property
    @check_instance_variable("image")
    def id(self) -> str:
        """
        Gets the unique ID of the Docker image.

        Returns:
            str: The Docker image ID.

        Example:
            >>> img = Image("myapp")
            >>> img.id
            'sha256:...'
        """
        return self.image.id

    @property
    @check_instance_variable("image")
    def architecture(self) -> str:
        """
        Gets the architecture of the Docker image.

        Returns:
            str: The image architecture (e.g., 'amd64').

        Example:
            >>> img = Image("myapp")
            >>> img.architecture
            'amd64'
        """
        return self.image.attrs["Architecture"]

    @property
    @check_instance_variable("image")
    def image_size(self) -> int:
        """
        Gets the size of the Docker image in bytes.

        Returns:
            int: The size of the image in bytes.

        Example:
            >>> img = Image("myapp")
            >>> img.image_size
            12345678
        """
        return int(self.image.attrs["Size"])

    @property
    @check_instance_variable("image")
    def labels(self) -> dict:
        """
        Gets the labels of the Docker image.

        Returns:
            dict: A dictionary of image labels.

        Example:
            >>> img = Image("myapp")
            >>> img.labels
            {'version': '1.0'}
        """
        return self.image.labels

    def get_fullname(self, tag: Optional[str] = None) -> str:
        """
        Constructs the full image name including the tag.

        Args:
            tag (Optional[str], optional): The tag to use; if not provided, the instance's tag is used.

        Returns:
            str: The full image name in the format "name:tag".

        Example:
            >>> img = Image("myapp", tag="v1")
            >>> img.get_fullname()
            'myapp:v1'
        """
        if tag is None:
            tag = self._tag
        return f"{self._image_name}:{tag}"

    def get_image(self, tag: Optional[str] = None) -> Optional[DockerImage]:
        """
        Retrieves the Docker image from the local repository.

        Args:
            tag (Optional[str], optional): The tag to use when retrieving the image. Defaults to the instance's tag.

        Returns:
            Optional[DockerImage]: The Docker image if found; otherwise, None.

        Example:
            >>> img = Image("myapp")
            >>> image_obj = img.get_image()
            >>> image_obj is not None  # Depends on local Docker images
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
            logger.error(f"Error accessing Docker API: {e}")
            return None

    def pull_image(self, tag: Optional[str] = None) -> Optional[DockerImage]:
        """
        Pulls the Docker image from a remote repository.

        Args:
            tag (Optional[str], optional): The tag to pull; defaults to the instance's tag if not provided.

        Returns:
            Optional[DockerImage]: The pulled Docker image if successful; otherwise, None.

        Example:
            >>> img = Image("myapp")
            >>> pulled = img.pull_image()
            >>> pulled is not None
        """
        tag = tag or self._tag
        logger.info(f"Pulling image {self.get_fullname(tag=tag)}")
        try:
            image = self._client.images.pull(self._image_name, tag=tag)
        except docker.errors.APIError as e:
            logger.error(f"Error pulling image {self.get_fullname(tag=tag)}")
        else:
            self._image = image
            return image

    def build_image(
        self,
        build_config: BuildConfig,
        dest: Union[str, Path],
        build_context: Optional[Union[str, Path]] = None,
    ) -> DockerImage:
        """
        Builds a Docker image using the specified build configuration.

        Args:
            build_config (BuildConfig): The configuration for building the image.
            dest (Union[str, Path]): The destination path where the Dockerfile will be saved.
            build_context (Optional[Union[str, Path]], optional): The build context directory. Defaults to build_config.context_dir.

        Returns:
            DockerImage: The built Docker image.

        Example:
            >>> build_config = BuildConfig()
            >>> img = Image("myapp")
            >>> built_img = img.build_image(build_config, "/tmp/dockerfile_dir")
            >>> built_img is not None
            True
        """
        build_context = build_context or build_config.context_dir
        build_context = Path(build_context)
        dest = Path(dest)
        if dest.is_dir():
            dest = dest.joinpath(build_config.docker_filename)
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
            logger.info(f"Image {image_name} built successfully!")
            self._image = image
        for line in build_log:
            logger.debug(line)
        return image

    def remove(
        self, tag: Optional[str] = None, force: bool = False, noprune: bool = False
    ):
        """
        Removes the Docker image from the local repository.

        Args:
            tag (Optional[str], optional): The tag to remove. Defaults to the instance's tag.
            force (bool, optional): Force removal. Defaults to False.
            noprune (bool, optional): Do not remove untagged parent images. Defaults to False.

        Returns:
            None

        Example:
            >>> img = Image("myapp")
            >>> img.remove()
        """
        image_name = self.get_fullname(tag=tag)
        args = {"image": image_name, "force": force, "noprune": noprune}
        try:
            logger.info(f"Removing image {image_name} with {args}")
            self._client.images.remove(**args)
        except docker.errors.APIError as e:
            logger.error(f"Failed to remove image {image_name}. Msg: {e}")
        finally:
            if self.exist:
                logger.error(f"Removing image {image_name} failed")
            else:
                logger.info(f"Removing image {image_name} succeeded")

    def info(self):
        """
        Prints detailed information about the Docker image.

        Returns:
            None

        Example:
            >>> img = Image("myapp")
            >>> img.info()
        """
        print(
            "\033[1;33m====================================== Image Info ===============================================\033[0m"
        )
        print(f"Name: {self.name}")
        print(f"Image ID: {self.id}")
        print(f"Architecture: {self.architecture}")
        print(f"Image Size: {format_memory(self.image_size)}")
        print(f"Labels: {self.labels}")


class ImageOrchestrator:
    """
    Orchestrates the building of multiple Docker images considering their dependencies.

    Attributes:
        _client (docker.DockerClient): The Docker client instance.
        _images (dict): A dictionary holding images and their build configuration and status.
    """

    def __init__(self, client: Optional[docker.DockerClient] = None):
        """
        Initializes the ImageOrchestrator.

        Args:
            client (Optional[docker.DockerClient], optional): The Docker client instance. Defaults to docker.from_env().

        Example:
            >>> orchestrator = ImageOrchestrator()
        """
        self._client = client or docker.from_env()
        self._images: Dict[str, Dict[str, Union[Optional[Image], BuildConfig, str]]] = (
            {}
        )

    @property
    def images(self) -> Dict[str, Dict[str, Union[Optional[Image], BuildConfig, str]]]:
        """
        Retrieves the dictionary of managed images.

        Returns:
            dict: A dictionary mapping image fullnames to their configuration and status.

        Example:
            >>> orch = ImageOrchestrator()
            >>> orch.images  # Initially empty dictionary
        """
        return self._images

    def add_image(self, image: Image, config: BuildConfig, base: Image = None):
        """
        Adds an image and its build configuration to the orchestrator.

        Args:
            image (Image): The Image instance to add.
            config (BuildConfig): The build configuration for the image.
            base (Image, optional): The base image that this image depends on, if any.

        Returns:
            bool: True if the image was added successfully.

        Example:
            >>> orch = ImageOrchestrator()
            >>> img = Image("myapp")
            >>> config = BuildConfig()
            >>> orch.add_image(img, config)
            True
        """
        assert isinstance(image, Image)
        assert isinstance(config, BuildConfig)
        assert image.get_image() not in self.images.keys()
        logger.info(f"Adding image {image.get_fullname()} to orchestrator")
        logger.info(f"{image.get_fullname()} image config: {config}")
        if base:
            assert isinstance(base, Image)
            assert base.get_fullname() in self.images.keys()
            logger.info(
                f"The image {image.get_fullname()} depends on base image {base.get_fullname()}"
            )
        self._images[image.get_fullname()] = {
            "image": image,
            "config": config,
            "base": base,
            "status": "init",
        }
        return image.get_fullname() in self._images.keys()

    def _topological_sort(self) -> list:
        """
        Performs a topological sort of images based on their dependency relationships.

        Returns:
            list: A list of image fullnames in the order they should be built.

        Raises:
            Exception: If a circular dependency is detected or a base image is not registered.

        Example:
            >>> sorted_images = orchestrator._topological_sort()
        """
        sorted_list = []
        temporary_marks = set()
        permanent_marks = set()

        def visit(image_key: str) -> None:
            if image_key in permanent_marks:
                return
            if image_key in temporary_marks:
                raise Exception(f"Circular dependency detected: {image_key}")
            temporary_marks.add(image_key)
            base = self._images[image_key].get("base")
            if base:
                base_key = base.get_fullname()
                if base_key in self._images:
                    visit(base_key)
                else:
                    raise Exception(f"Base image {base_key} is not registered")
            permanent_marks.add(image_key)
            sorted_list.append(image_key)

        for key in self._images:
            if key not in permanent_marks:
                visit(key)
        logger.debug(f"After topological sort: {sorted_list}")
        return sorted_list

    def build_all(self):
        """
        Builds all registered images in the correct order based on dependencies.

        Returns:
            None

        Example:
            >>> orchestrator.build_all()
        """
        build_sorted_list = self._topological_sort()
        tmp_dir = Path(
            get_temp_dir_with_specific_path(f"Bulk-Image-Build-{unique_id()}")
        )
        logger.info(f"Building images in temporary directory: {tmp_dir}")
        for image_key in tqdm(build_sorted_list):
            image: Image = self._images[image_key]["image"]
            config: BuildConfig = self._images[image_key]["config"]
            base: Optional[Image] = self._images[image_key]["base"]
            logger.debug(f"starting build for {image.get_fullname()}")
            if base is not None:
                logger.debug(f"Original config: {config}")
                base_config: BuildConfig = self._images[base.get_fullname()]["config"]
                config.base_image = base.get_fullname()
                config.python_deps_manager = base_config.python_deps_manager
                config.sys_deps_manager = base_config.sys_deps_manager
                config.user = base_config.user
                config.uid = base_config.uid
                config.add_label("BaseImage", base.get_fullname())
                logger.debug(f"Config after inheritance: {config}")

            target_dir = tmp_dir.joinpath(image.get_fullname().replace(":", "-"))
            logger.info(f"The Dockerfile will be saved to {target_dir}")
            image.build_image(build_config=config, dest=target_dir)
            if not image.exist:
                self.images[image_key]["status"] = "failed"
                raise RuntimeError(f"Failed to build image {image.get_fullname()}")
            else:
                self._images[image_key]["status"] = "success"
