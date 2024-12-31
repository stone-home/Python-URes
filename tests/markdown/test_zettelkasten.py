import pytest
import os
from ures.markdown.zettelkasten import (
    Zettelkasten,
)  # Adjust based on your project structure
import frontmatter


def test_zettelkasten_initialization(valid_zettelkasten):
    zk = valid_zettelkasten
    assert zk.title == "Sample Title"
    assert zk.type == "permanent"
    assert zk.url == "http://example.com"
    assert zk.tags == ["sample", "zettelkasten"]
    assert zk.aliases == ["sample-alias"]
    assert zk.metadata["id"] == "unique-id-12345"  # Mocked value
    assert zk.metadata["create"] == "2024-12-31T23:59:59Z"  # Mocked value


def test_zettelkasten_initialization_missing_title(mock_time_now, mock_unique_id):
    with pytest.raises(ValueError) as excinfo:
        Zettelkasten(
            title=None,  # Invalid title
            n_type="permanent",
            url="http://example.com",
            tags=["sample", "zettelkasten"],
            aliases=["sample-alias"],
        )
    assert "Title must be a non-empty string." in str(excinfo.value)


def test_zettelkasten_initialization_empty_title(mock_time_now, mock_unique_id):
    with pytest.raises(ValueError) as excinfo:
        Zettelkasten(
            title="",  # Empty title
            n_type="permanent",
            url="http://example.com",
            tags=["sample", "zettelkasten"],
            aliases=["sample-alias"],
        )
    assert "Title must be a non-empty string." in str(excinfo.value)


def test_zettelkasten_initialization_invalid_type(mock_time_now, mock_unique_id):
    with pytest.raises(ValueError) as excinfo:
        Zettelkasten(
            title="Valid Title",
            n_type="invalid-type",  # Invalid type
            url="http://example.com",
            tags=["sample", "zettelkasten"],
            aliases=["sample-alias"],
        )
    assert (
        "Invalid type 'invalid-type'. Allowed types are: fleeting, literature, permanent, atom."
        in str(excinfo.value)
    )


def test_zettelkasten_initialization_invalid_url(mock_time_now, mock_unique_id):
    with pytest.raises(ValueError) as excinfo:
        Zettelkasten(
            title="Valid Title",
            n_type="permanent",
            url=123,  # Invalid URL type
            tags=["sample", "zettelkasten"],
            aliases=["sample-alias"],
        )
    assert "URL must be a string." in str(excinfo.value)


def test_zettelkasten_initialization_invalid_tags(mock_time_now, mock_unique_id):
    with pytest.raises(ValueError) as excinfo:
        Zettelkasten(
            title="Valid Title",
            n_type="permanent",
            url="http://example.com",
            tags="not-a-list",  # Invalid tags type
            aliases=["sample-alias"],
        )
    assert "Tags must be a list." in str(excinfo.value)


def test_zettelkasten_initialization_invalid_aliases(mock_time_now, mock_unique_id):
    with pytest.raises(ValueError) as excinfo:
        Zettelkasten(
            title="Valid Title",
            n_type="permanent",
            url="http://example.com",
            tags=["sample", "zettelkasten"],
            aliases="not-a-list",  # Invalid aliases type
        )
    assert "Aliases must be a list." in str(excinfo.value)


def test_zettelkasten_title_property(valid_zettelkasten):
    zk = valid_zettelkasten
    assert zk.title == "Sample Title"
    zk.title = "Updated Title"
    assert zk.title == "Updated Title"
    assert zk.metadata["title"] == "Updated Title"

    with pytest.raises(ValueError) as excinfo:
        zk.title = ""
    assert "Title must be a non-empty string." in str(excinfo.value)


def test_zettelkasten_type_property(valid_zettelkasten):
    zk = valid_zettelkasten
    assert zk.type == "permanent"
    zk.type = "atom"
    assert zk.type == "atom"
    assert zk.metadata["type"] == "atom"

    with pytest.raises(ValueError) as excinfo:
        zk.type = "invalid-type"
    assert (
        "Invalid type 'invalid-type'. Allowed types are: fleeting, literature, permanent, atom."
        in str(excinfo.value)
    )


def test_zettelkasten_url_property(valid_zettelkasten):
    zk = valid_zettelkasten
    assert zk.url == "http://example.com"
    zk.url = "http://newurl.com"
    assert zk.url == "http://newurl.com"
    assert zk.metadata["url"] == "http://newurl.com"

    with pytest.raises(ValueError) as excinfo:
        zk.url = 123  # Invalid URL type
    assert "URL must be a string." in str(excinfo.value)


def test_zettelkasten_tags_property(valid_zettelkasten):
    zk = valid_zettelkasten
    assert zk.tags == ["sample", "zettelkasten"]
    zk.tags = ["updated-tag"]
    assert zk.tags == ["updated-tag"]
    assert zk.metadata["tags"] == ["updated-tag"]

    with pytest.raises(ValueError) as excinfo:
        zk.tags = "not-a-list"
    assert "Tags must be a list." in str(excinfo.value)


def test_zettelkasten_aliases_property(valid_zettelkasten):
    zk = valid_zettelkasten
    assert zk.aliases == ["sample-alias"]
    zk.aliases = ["updated-alias"]
    assert zk.aliases == ["updated-alias"]
    assert zk.metadata["aliases"] == ["updated-alias"]

    with pytest.raises(ValueError) as excinfo:
        zk.aliases = "not-a-list"
    assert "Aliases must be a list." in str(excinfo.value)


def test_zettelkasten_add_tag(valid_zettelkasten):
    zk = valid_zettelkasten
    zk.add_tag("new-tag")
    assert zk.tags == ["sample", "zettelkasten", "new-tag"]
    assert zk.metadata["tags"] == ["sample", "zettelkasten", "new-tag"]


def test_zettelkasten_remove_tag(valid_zettelkasten):
    zk = valid_zettelkasten
    zk.remove_tag("sample")
    assert zk.tags == ["zettelkasten"]
    assert zk.metadata["tags"] == ["zettelkasten"]

    # Attempting to remove a non-existent tag should raise ValueError
    with pytest.raises(ValueError):
        zk.remove_tag("non-existent-tag")


def test_zettelkasten_add_alias(valid_zettelkasten):
    zk = valid_zettelkasten
    zk.add_alias("new-alias")
    assert zk.aliases == ["sample-alias", "new-alias"]
    assert zk.metadata["aliases"] == ["sample-alias", "new-alias"]


def test_zettelkasten_remove_alias(valid_zettelkasten):
    zk = valid_zettelkasten
    zk.remove_alias("sample-alias")
    assert zk.aliases == []
    assert zk.metadata["aliases"] == []

    # Attempting to remove a non-existent alias should raise ValueError
    with pytest.raises(ValueError):
        zk.remove_alias("non-existent-alias")


def test_zettelkasten_validation_after_setting(valid_zettelkasten):
    zk = valid_zettelkasten
    # Remove a mandatory field 'title' to trigger validation
    zk.remove_frontmatter("title")
    with pytest.raises(ValueError) as excinfo:
        zk.validation_frontmatter()
    assert "Missing mandatory front matter fields: title" in str(excinfo.value)


def test_zettelkasten_validation_during_save(valid_zettelkasten, tmp_path):
    zk = valid_zettelkasten
    # Remove a mandatory field 'type'
    zk.remove_frontmatter("type")

    # Attempting to save should raise ValueError
    with pytest.raises(ValueError) as excinfo:
        zk.save(str(tmp_path / "invalid_save.md"))
    assert "Missing mandatory front matter fields: type" in str(excinfo.value)

    # Ensure the file was not created
    assert not os.path.exists(tmp_path / "invalid_save.md")


def test_zettelkasten_to_markdown(valid_zettelkasten):
    zk = valid_zettelkasten
    markdown_str = zk.to_markdown()

    expected_front_matter = frontmatter.dumps(
        frontmatter.Post(
            zk.content,
            title=zk.title,
            type=zk.type,
            url=zk.url,
            create=zk.metadata["create"],
            id=zk.metadata["id"],
            tags=zk.tags,
            aliases=zk.aliases,
        )
    )

    assert markdown_str == expected_front_matter


def test_zettelkasten_save_and_load(valid_zettelkasten, tmp_path):
    zk = valid_zettelkasten
    file_path = tmp_path / "zettelkasten.md"

    # Save the document
    zk.save(str(file_path))
    assert os.path.isfile(file_path)

    # Load the document
    loaded_zk = Zettelkasten.from_file(str(file_path))

    assert loaded_zk.title == zk.title
    assert loaded_zk.type == zk.type
    assert loaded_zk.url == zk.url
    assert loaded_zk.tags == zk.tags
    assert loaded_zk.aliases == zk.aliases
    assert loaded_zk.metadata["id"] == zk.metadata["id"]
    assert loaded_zk.metadata["create"] == zk.metadata["create"]
    assert loaded_zk.content == zk.content


def test_zettelkasten_load_missing_mandatory_fields(valid_zettelkasten, tmp_path):
    file_path = tmp_path / "missing_fields.md"
    content = "---\ntype: permanent\nurl: http://example.com\ncreate: 2024-12-31T23:59:59Z\nid: unique-id-12345\ntags:\n  - sample\naliases:\n  - alias1\n---\n\n# Heading"  # Missing 'title'
    file_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        Zettelkasten.from_file(str(file_path))
    assert "Title must be a non-empty string." in str(excinfo.value)


def test_zettelkasten_add_duplicate_tags(valid_zettelkasten):
    zk = valid_zettelkasten
    zk.add_tag("sample")  # 'sample' already exists
    assert zk.tags == [
        "sample",
        "zettelkasten",
        "sample",
    ]  # No duplicate due to check in add_tag
    assert zk.metadata["tags"] == ["sample", "zettelkasten", "sample"]  # No duplicate


def test_zettelkasten_add_duplicate_aliases(valid_zettelkasten):
    zk = valid_zettelkasten
    zk.add_alias("sample-alias")  # 'sample-alias' already exists
    assert zk.aliases == [
        "sample-alias",
        "sample-alias",
    ]  # No duplicate due to check in add_alias
    assert zk.metadata["aliases"] == ["sample-alias", "sample-alias"]  # No duplicate


def test_zettelkasten_remove_nonexistent_tag(valid_zettelkasten):
    zk = valid_zettelkasten
    with pytest.raises(ValueError):
        zk.remove_tag("nonexistent-tag")


def test_zettelkasten_remove_nonexistent_alias(valid_zettelkasten):
    zk = valid_zettelkasten
    with pytest.raises(ValueError):
        zk.remove_alias("nonexistent-alias")


def test_zettelkasten_clear_frontmatter(valid_zettelkasten):
    # Attempting to clear front matter should raise ValueError because mandatory fields are missing
    valid_zettelkasten.clear_frontmatter()
    assert valid_zettelkasten.metadata == {}
