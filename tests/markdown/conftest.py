import pytest
from ures.markdown import Zettelkasten


@pytest.fixture
def mock_time_now(monkeypatch):
    mock_time = "2024-12-31T23:59:59Z"
    # Patch 'time_now' in the 'ures.markdown.zettelkasten' module's namespace
    monkeypatch.setattr("ures.markdown.zettelkasten.time_now", lambda: mock_time)
    return mock_time


@pytest.fixture
def mock_unique_id(monkeypatch):
    mock_id = "unique-id-12345"
    # Patch 'unique_id' in the 'ures.markdown.zettelkasten' module's namespace
    monkeypatch.setattr("ures.markdown.zettelkasten.unique_id", lambda: mock_id)
    return mock_id


@pytest.fixture
def valid_zettelkasten(mock_time_now, mock_unique_id):
    zk = Zettelkasten(
        title="Sample Title",
        n_type="permanent",
        url="http://example.com",
        tags=["sample", "zettelkasten"],
        aliases=["sample-alias"],
    )
    zk.add_content("This is a sample content.")
    zk.add_content("This is another sample content.")
    return zk


@pytest.fixture
def invalid_zettelkasten_missing_fields(mock_time_now, mock_unique_id):
    # Missing 'title'
    with pytest.raises(ValueError) as excinfo:
        Zettelkasten(
            title="",
            n_type="permanent",
            url="http://example.com",
            tags=["sample", "zettelkasten"],
            aliases=["sample-alias"],
        )
    return excinfo.value
