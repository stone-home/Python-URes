import time
from pathlib import Path
import tempfile
import pytest
from ures.docker.runtime import SimpleRuntime


# Dummy container class to simulate Container behavior.
class DummyContainer:
    def __init__(self, image_name, run_success=True, log_content="dummy log content"):
        """
        Initializes a dummy container for testing.

        Args:
            image_name (str): The name of the container.
            run_success (bool, optional): If True, run() will set the container to running.
                If False, the container will remain not running.
            log_content (str, optional): The log content to return when logs() is called.
        """
        self.image_name = image_name
        self._run_success = run_success
        self._running = False
        self.wait_called = False
        self._stopped = False
        self._removed = False
        self._log = log_content
        self.is_created = True

    def run(self):
        """
        Simulates running the container. Sets the container to running if _run_success is True.
        """
        if self._run_success:
            self._running = True
        else:
            self._running = False

    @property
    def is_running(self):
        """
        Indicates whether the container is running.

        Returns:
            bool: True if running, False otherwise.
        """
        return self._running

    def wait(self):
        """
        Simulates waiting for the container to finish execution.
        """
        self.wait_called = True

    def stop(self):
        """
        Simulates stopping the container.
        """
        self._stopped = True
        self._running = False

    def remove(self):
        """
        Simulates removing the container.
        """
        self._removed = True
        self.is_created = False

    def logs(self):
        """
        Returns the simulated log content.

        Returns:
            str: The dummy log content.
        """
        return self._log


class TestSimpleRuntime:
    def test_run_success(self):
        """
        Test that SimpleRuntime.run successfully runs a container that becomes "running"
        and calls wait().

        Example:
            >>> container.is_running is True
            True
            >>> container.wait_called is True
        """
        container = DummyContainer("dummy:latest", run_success=True)
        runtime = SimpleRuntime([container])
        runtime.run()
        assert container.is_running is True
        assert container.wait_called is True

    def test_run_failure(self):
        """
        Test that SimpleRuntime.run logs an error and does not call wait() for a container
        that fails to become "running".

        Example:
            >>> container.is_running is False
            True
            >>> container.wait_called is False
        """
        container = DummyContainer("dummy:latest", run_success=False)
        runtime = SimpleRuntime([container])
        runtime.run()
        assert container.is_running is False
        assert container.wait_called is False

    def test_stop(self):
        """
        Test that SimpleRuntime.stop calls the stop() method on each container.

        Example:
            >>> container._stopped is True
            True
        """
        container = DummyContainer("dummy:latest", run_success=True)
        # Manually set running state to True for testing stop
        container._running = True
        runtime = SimpleRuntime([container])
        runtime.stop()
        assert container._stopped is True
        assert container.is_running is False

    def test_remove(self):
        """
        Test that SimpleRuntime.remove calls the remove() method on each container,
        causing is_created to become False.

        Example:
            >>> container.is_created is False
            True
        """
        container = DummyContainer("dummy:latest", run_success=True)
        runtime = SimpleRuntime([container])
        runtime.remove()
        assert container._removed is True
        assert container.is_created is False

    def test_logs(self, tmp_path):
        """
        Test that SimpleRuntime.logs writes the container logs to a file in the correct directory.

        Args:
            tmp_path (Path): Temporary directory fixture provided by pytest.

        Example:
            >>> log_file.read_text() == "dummy log content"
        """
        container = DummyContainer(
            "dummy:latest",
            run_success=True,
            log_content="dummy log content".encode(encoding="utf-8"),
        )
        runtime = SimpleRuntime([container])
        output_dir = tmp_path / "logs_test"
        output_dir.mkdir()
        runtime.logs(output_dir)
        expected_dir = output_dir / container.image_name.replace(":", "-")
        log_file = expected_dir / "logs.txt"
        assert log_file.exists()
        content = log_file.read_text()
        assert content == "dummy log content"
