import logging
import tqdm
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, List
from ures.docker import Container

logger = logging.getLogger(__name__)


class Runtime(ABC):
    def __init__(self, containers: List[Container], interval: float = 0.5):
        """
        Initializes the Runtime with a list of containers.

        Args:
            containers (List[Container]): A list of Container instances to manage.
                Each container must have been created (i.e. container.is_created is True).

        Example:
            >>> runtime = SomeRuntime([container1, container2])
        """
        assert all([container.is_created for container in containers]) is True
        self._containers: List[Container] = containers
        self._interval = interval

    def _regular_delay(self):
        time.sleep(self._interval)

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Run all managed containers.

        Returns:
            None

        Example:
            >>> runtime.run()
        """
        pass

    @abstractmethod
    def stop(self, *args, **kwargs):
        """
        Stop all managed containers.

        Returns:
            None

        Example:
            >>> runtime.stop()
        """
        pass

    @abstractmethod
    def remove(self, *args, **kwargs):
        """
        Remove all managed containers.

        Returns:
            None

        Example:
            >>> runtime.remove()
        """
        pass

    @abstractmethod
    def logs(self, output_dir: Union[str, Path], *args, **kwargs):
        """
        Retrieve logs from all managed containers and save them to the specified directory.

        Args:
            output_dir (Union[str, Path]): The directory where log files should be saved.

        Returns:
            None

        Example:
            >>> runtime.logs("/tmp/container_logs")
        """
        pass


class SimpleRuntime(Runtime):
    def run(self, *args, **kwargs):
        """
        Runs each container by calling its run() method. If the container becomes running,
        wait() is called; otherwise, an error is logged.

        Returns:
            None

        Example:
            >>> runtime = SimpleRuntime([container1, container2])
            >>> runtime.run()
        """
        for container in tqdm.tqdm(self._containers):
            container.run()
            self._regular_delay()
            if container.is_running is False:
                logging.error(f"[Failed] {container.image_name} failed to start")
                continue
            else:
                container.wait()

    def stop(self, *args, **kwargs):
        """
        Stops each container by calling its stop() method.

        Returns:
            None

        Example:
            >>> runtime.stop()
        """
        for container in tqdm.tqdm(self._containers):
            container.stop()
            self._regular_delay()

    def remove(self, *args, **kwargs):
        """
        Removes each container by calling its remove() method.

        Returns:
            None

        Example:
            >>> runtime.remove()
        """
        for container in tqdm.tqdm(self._containers):
            container.remove()
            self._regular_delay()

    def logs(self, output_dir: Union[str, Path], *args, **kwargs):
        """
        Retrieves logs from each container and writes them to a "logs.txt" file in a directory
        named after the container's image (with ":" replaced by "-").

        Args:
            output_dir (Union[str, Path]): The directory where logs should be stored.

        Returns:
            None

        Example:
            >>> runtime.logs("/tmp/container_logs")
        """
        output_dir = Path(output_dir)
        for container in tqdm.tqdm(self._containers):
            container_dir = output_dir / container.image_name.replace(":", "-")
            container_dir.mkdir(
                parents=True, exist_ok=True
            )  # Ensure the directory exists
            log = container.logs()
            with open(container_dir / "logs.txt", "w") as f:
                f.write(log.decode("utf-8"))
