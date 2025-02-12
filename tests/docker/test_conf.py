import pytest
from pathlib import Path
from ures.docker.conf import BuildConfig  # Assuming your code is in build_config.py


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
