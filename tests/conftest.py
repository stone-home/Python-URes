import os
import uuid
import pytest


@pytest.fixture(scope="function")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp(str(uuid.uuid4()))


@pytest.fixture(scope="class")
def root_dir():
    return os.path.dirname(os.path.abspath(__file__))
