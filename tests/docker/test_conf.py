import pytest
from pathlib import Path
from ures.docker.conf import (
    BuildConfig,
    RuntimeConfig,
)  # Assuming your code is in build_config.py


class TestBuildConfig:
    def test_defaults(self):
        config = BuildConfig()
        assert config.base_image == "python:3.10-slim"
        assert config.platform is None
        assert config.python_deps_manager == "pip"
        assert config.python_dependencies is None
        assert config.sys_deps_manager == "apt"
        assert config.sys_dependencies is None
        assert config.labels is None
        assert config.uid is None
        assert config.user is None
        assert config.entrypoint is None
        assert config.cmd is None
        assert config.environment is None
        assert config.copies is None
        assert config.context_dir == Path.cwd()

    def test_add_label(self):
        config = BuildConfig()
        config.add_label("maintainer", "test@example.com")
        assert config.labels == [("maintainer", "test@example.com")]

    def test_add_copy(self):
        config = BuildConfig()
        config.add_copy("./app", "/app")
        assert config.copies == [{"src": "./app", "dest": "/app"}]

    def test_add_environment(self):
        config = BuildConfig()
        config.add_environment("DEBUG", "true")
        assert config.environment == {"DEBUG": "true"}

    def test_set_context_dir_str(self):
        config = BuildConfig()
        test_dir = Path("./test_dir_str")  # Unique name to avoid conflicts
        test_dir.mkdir(exist_ok=True)
        config.set_context_dir(str(test_dir))
        assert config.context_dir == test_dir
        test_dir.rmdir()

    def test_set_context_dir_path(self):
        config = BuildConfig()
        test_dir = Path("./test_dir_path")  # Unique name
        test_dir.mkdir(exist_ok=True)
        config.set_context_dir(test_dir)
        assert config.context_dir == test_dir
        test_dir.rmdir()

    def test_set_context_dir_invalid(self):
        config = BuildConfig()
        with pytest.raises(ValueError):
            config.set_context_dir("./nonexistent_dir")

    def test_add_python_dependency(self):
        config = BuildConfig()
        config.add_python_dependency("requests")
        assert config.python_dependencies == ["requests"]

    def test_add_system_dependency(self):
        config = BuildConfig()
        config.add_system_dependency("build-essential")
        assert config.sys_dependencies == ["build-essential"]

    def test_set_entrypoint_str(self):
        config = BuildConfig()
        config.set_entrypoint("/app/run.sh")
        assert config.entrypoint == ["/app/run.sh"]

    def test_set_entrypoint_list(self):
        config = BuildConfig()
        config.set_entrypoint(["/app/run.sh", "--arg1"])
        assert config.entrypoint == ["/app/run.sh", "--arg1"]

    def test_set_cmd_str(self):
        config = BuildConfig()
        config.set_cmd("--help")
        assert config.cmd == ["--help"]

    def test_set_cmd_list(self):
        config = BuildConfig()
        config.set_cmd(["run", "--verbose"])
        assert config.cmd == ["run", "--verbose"]

    def test_all_features(self):  # Integration test
        config = BuildConfig()
        config.add_label("version", "1.0")
        config.add_copy("./my_app", "/app")
        config.add_environment("API_KEY", "secret")
        test_dir = Path("./test_dir_integration")
        test_dir.mkdir(exist_ok=True)
        config.set_context_dir(test_dir)
        config.add_python_dependency("numpy")
        config.add_system_dependency("libpq-dev")
        config.set_entrypoint(["python", "/app/main.py"])
        config.set_cmd(["--config", "/app/config.ini"])

        assert config.labels == [("version", "1.0")]
        assert config.copies == [{"src": "./my_app", "dest": "/app"}]
        assert config.environment == {"API_KEY": "secret"}
        assert config.context_dir == test_dir
        assert config.python_dependencies == ["numpy"]
        assert config.sys_dependencies == ["libpq-dev"]
        assert config.entrypoint == ["python", "/app/main.py"]
        assert config.cmd == ["--config", "/app/config.ini"]
        test_dir.rmdir()


class TestRuntimeConfig:
    def test_defaults(self):
        config = RuntimeConfig()
        assert config.image_name == "model-runner"
        assert config.name is None
        assert config.platform is None
        assert config.detach is True
        assert config.user is None
        assert config.remove is False
        assert config.cpus is None
        assert config.gpus is None
        assert config.memory is None
        assert config.entrypoint is None
        assert config.command is None
        assert config.env is None
        assert config.volumes is None
        assert config.subnet is None
        assert config.ipv4 is None
        assert config.network_mode == "bridge"
        config.out_dir.mkdir(exist_ok=True, parents=True)
        assert config.out_dir.exists()

    def test_custom_values(self):
        custom_config = RuntimeConfig(
            image_name="custom-image",
            name="custom-container",
            platform="linux/amd64",
            detach=False,
            user="testuser",
            remove=True,
            cpus=4,
            gpus=["0", "1"],
            memory="4g",
            entrypoint=["/bin/sh"],
            command=["-c", "echo hello"],
            env={"VAR1": "value1"},
            volumes={"/host/path": {"bind": "/container/path", "mode": "rw"}},
            subnet="192.168.1.0/24",
            ipv4="192.168.1.100",
            network_mode="host",
            out_dir=Path("/tmp/runtime_out"),
        )

        assert custom_config.image_name == "custom-image"
        assert custom_config.name == "custom-container"
        assert custom_config.platform == "linux/amd64"
        assert custom_config.detach is False
        assert custom_config.user == "testuser"
        assert custom_config.remove is True
        assert custom_config.cpus == 4
        assert custom_config.gpus == ["0", "1"]
        assert custom_config.memory == "4g"
        assert custom_config.entrypoint == ["/bin/sh"]
        assert custom_config.command == ["-c", "echo hello"]
        assert custom_config.env == {"VAR1": "value1"}
        assert custom_config.volumes == {
            "/host/path": {"bind": "/container/path", "mode": "rw"}
        }
        assert custom_config.subnet == "192.168.1.0/24"
        assert custom_config.ipv4 == "192.168.1.100"
        assert custom_config.network_mode == "host"
        assert custom_config.out_dir == Path("/tmp/runtime_out")

    def test_invalid_values(self):
        with pytest.raises(ValueError):
            RuntimeConfig(cpus="four")  # Invalid type for cpus
        with pytest.raises(ValueError):
            RuntimeConfig(memory=1024)  # Invalid type for memory
        with pytest.raises(ValueError):
            RuntimeConfig(gpus="gpu0")  # Invalid type for gpus

    def test_methods(self):
        config = RuntimeConfig()
        config.add_env("NEW_VAR", "new_value")
        assert config.env == {"NEW_VAR": "new_value"}

        config.add_volume("/host/dir", "/container/dir")
        assert config.volumes == {"/host/dir": {"bind": "/container/dir", "mode": "rw"}}
