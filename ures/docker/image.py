import logging
import docker
from typing import Optional
from docker.models.images import Image as DockerImage

logger = logging.getLogger(__name__)



class Image:
    def __init__(self, image_name: str):
        self._image_name = image_name
        self._client = docker.from_env()

    @property
    def exist(self) -> bool:
        try:
            self._client.images.get(f"{image_name}:{tag}")
            return True
        except docker.errors.ImageNotFound:
            return False
        except docker.errors.APIError as e:
            print(f"Error accessing Docker API: {e}")
            return False

    def get_image(self, tag: Optional[str] = None) -> str:
        _image_name = self._image_name
        _name, _tag = self.get_image_name(_image_name)
        if tag is not None:
            _tag = tag
        _search_results = self._client.images.list(name=f"{_name}:{_tag}")
        if len(_search_results) > 0:
            logger.info(f"{len(_research)} images found for {_name}:{_tag}")
            logger.info(f"The 1st image is {_research[0]} picked")
            return ImageInfo(_research[0])
        logger.info(f"Image {_image_name} is not found")
        return None

    def get_image_name(self, name: Optional[str] = None) -> tuple[str, str]:
        """Get the image name and tag

        Returns:
            tuple[str, str]: The image name and tag

        """
        _image_name = name or self._image_name
        logger.debug(
            f"Getting image name and tag for {_image_name}, input name: {name}"
        )
        _split_image_name = str(_image_name).split(":")
        if len(_split_image_name) == 1:
            return _split_image_name[0], "latest"
        return _split_image_name[0], _split_image_name[1]


