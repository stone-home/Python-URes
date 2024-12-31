import pytest
from ures.markdown import MarkdownDocument
import frontmatter


def test_init_without_metadata():
    doc = MarkdownDocument(content="# Hello World")
    assert doc.content == "# Hello World"
    assert doc.metadata == {}


def test_init_with_metadata():
    metadata = {"title": "Greeting", "tags": ["intro", "welcome"]}
    doc = MarkdownDocument(content="# Hello World", metadata=metadata)
    assert doc.content == "# Hello World"
    assert doc.metadata == metadata


def test_from_file_success(tmp_path):
    file_path = tmp_path / "test.md"
    content = (
        "---\ntitle: Test Document\ntags:\n  - test\n  - markdown\n---\n\n# Heading"
    )
    file_path.write_text(content, encoding="utf-8")

    doc = MarkdownDocument.from_file(str(file_path))
    assert doc.content == "# Heading"
    assert doc.metadata == {"title": "Test Document", "tags": ["test", "markdown"]}


def test_from_file_not_found():
    with pytest.raises(FileNotFoundError):
        MarkdownDocument.from_file("non_existent.md")


def test_add_content_append():
    doc = MarkdownDocument(content="# Initial Heading")
    doc.add_content("This is additional content.", append=True)
    assert doc.content == "# Initial Heading\nThis is additional content."


def test_add_content_prepend():
    doc = MarkdownDocument(content="# Initial Heading")
    doc.add_content("This is prepended content.", append=False)
    assert doc.content == "This is prepended content.\n# Initial Heading"


def test_add_content_empty_initial():
    doc = MarkdownDocument()
    doc.add_content("First content.", append=True)
    assert doc.content == "First content."

    doc.add_content("Second content.", append=True)
    assert doc.content == "First content.\nSecond content."

    doc.add_content("Prepended content.", append=False)
    assert doc.content == "Prepended content.\nFirst content.\nSecond content."


def test_set_frontmatter_simple():
    doc = MarkdownDocument()
    doc.set_frontmatter("title", "Simple Title")
    assert doc.get_frontmatter("title") == "Simple Title"


def test_set_frontmatter_nested_dict():
    doc = MarkdownDocument()
    doc.set_frontmatter("author.name", "John Doe")
    doc.set_frontmatter("author.contact.email", "john@example.com")
    assert doc.get_frontmatter("author.name") == "John Doe"
    assert doc.get_frontmatter("author.contact.email") == "john@example.com"


def test_set_frontmatter_nested_list(tmp_path):
    doc = MarkdownDocument()
    doc.set_frontmatter("sections.0.title", "Introduction")
    doc.set_frontmatter("sections.0.content", "Welcome to the introduction.")
    doc.set_frontmatter("sections.1.title", "Conclusion")
    doc.set_frontmatter("sections.1.content", "Wrapping up.")

    assert doc.get_frontmatter("sections.0.title") == "Introduction"
    assert doc.get_frontmatter("sections.0.content") == "Welcome to the introduction."
    assert doc.get_frontmatter("sections.1.title") == "Conclusion"
    assert doc.get_frontmatter("sections.1.content") == "Wrapping up."


def test_set_frontmatter_with_lists():
    doc = MarkdownDocument()
    doc.set_frontmatter("tags", ["python", "markdown"], overwrite=True)
    assert doc.get_frontmatter("tags") == ["python", "markdown"]

    doc.set_frontmatter("tags", "frontmatter", overwrite=False)
    assert doc.get_frontmatter("tags") == ["python", "markdown", "frontmatter"]


def test_set_frontmatter_overwrite():
    doc = MarkdownDocument()
    doc.set_frontmatter("title", "Original Title")
    doc.set_frontmatter("title", "New Title", overwrite=True)
    assert doc.get_frontmatter("title") == "New Title"


def test_set_frontmatter_append_to_list():
    doc = MarkdownDocument()
    doc.set_frontmatter("keywords", ["test"], overwrite=False)
    doc.set_frontmatter("keywords", "unit-test", overwrite=False)
    assert doc.get_frontmatter("keywords") == ["test", "unit-test"]


def test_set_frontmatter_mixed_nested():
    doc = MarkdownDocument()
    doc.set_frontmatter("authors.0.name", "Alice")
    doc.set_frontmatter("authors.0.contact.email", "alice@example.com")
    doc.set_frontmatter("authors.1.name", "Bob")
    doc.set_frontmatter("authors.1.contact.email", "bob@example.com")

    assert doc.get_frontmatter("authors.0.name") == "Alice"
    assert doc.get_frontmatter("authors.0.contact.email") == "alice@example.com"
    assert doc.get_frontmatter("authors.1.name") == "Bob"
    assert doc.get_frontmatter("authors.1.contact.email") == "bob@example.com"


def test_get_frontmatter_non_existent():
    doc = MarkdownDocument()
    assert doc.get_frontmatter("nonexistent") is None


def test_get_frontmatter_nested():
    metadata = {
        "author": {"name": "John Doe", "contact": {"email": "john@example.com"}},
        "sections": [
            {"title": "Intro", "content": "Welcome."},
            {"title": "Conclusion", "content": "Goodbye."},
        ],
    }
    doc = MarkdownDocument(metadata=metadata)
    assert doc.get_frontmatter("author.name") == "John Doe"
    assert doc.get_frontmatter("author.contact.email") == "john@example.com"
    assert doc.get_frontmatter("sections.0.title") == "Intro"
    assert doc.get_frontmatter("sections.1.content") == "Goodbye."
    assert doc.get_frontmatter("sections.2.title") is None


def test_remove_frontmatter_simple():
    doc = MarkdownDocument()
    doc.set_frontmatter("title", "Removable Title")
    doc.remove_frontmatter("title")
    assert doc.get_frontmatter("title") is None


def test_remove_frontmatter_nested():
    metadata = {
        "author": {"name": "John Doe", "contact": {"email": "john@example.com"}}
    }
    doc = MarkdownDocument(metadata=metadata)
    doc.remove_frontmatter("author.contact.email")
    assert doc.get_frontmatter("author.contact.email") is None
    assert doc.get_frontmatter("author.contact") == {}


def test_remove_frontmatter_list_item():
    metadata = {
        "sections": [
            {"title": "Intro", "content": "Welcome."},
            {"title": "Conclusion", "content": "Goodbye."},
        ]
    }
    doc = MarkdownDocument(metadata=metadata)
    doc.remove_frontmatter("sections.0.title")
    assert doc.get_frontmatter("sections.0.title") is None
    doc.remove_frontmatter("sections.1")
    assert len(doc.get_frontmatter("sections")) == 1
    assert doc.get_frontmatter("sections.0.title") is None


def test_to_markdown():
    metadata = {
        "title": "Serialization Test",
        "tags": ["serialize", "test"],
        "author": {"name": "Jane Doe", "contact": {"email": "jane@example.com"}},
    }
    content = "# Heading\n\nSome content here."
    doc = MarkdownDocument(content=content, metadata=metadata)
    markdown_str = doc.to_markdown()

    expected_front_matter = frontmatter.dumps(frontmatter.Post(content, **metadata))
    assert markdown_str == expected_front_matter


def test_save(tmp_path):
    file_path = tmp_path / "saved.md"
    metadata = {
        "title": "Save Test",
        "tags": ["save", "test"],
        "author": {"name": "Sam Smith", "contact": {"email": "sam@example.com"}},
    }
    content = "# Saved Heading\n\nSaved content."
    doc = MarkdownDocument(content=content, metadata=metadata)
    doc.save(str(file_path))

    saved_content = file_path.read_text(encoding="utf-8")
    expected_content = frontmatter.dumps(frontmatter.Post(content, **metadata))
    assert saved_content == expected_content


def test_load_from_file(tmp_path):
    file_path = tmp_path / "load_test.md"
    metadata = {
        "title": "Load Test",
        "tags": ["load", "test"],
        "author": {"name": "Laura Lee", "contact": {"email": "laura@example.com"}},
    }
    content = "# Load Heading\n\nLoad content here."
    file_content = frontmatter.dumps(frontmatter.Post(content, **metadata))
    file_path.write_text(file_content, encoding="utf-8")

    doc = MarkdownDocument()
    doc.load_from_file(str(file_path))

    assert doc.content == content
    assert doc.metadata == metadata


def test_clear_content():
    metadata = {"title": "Clear Content Test"}
    content = "# Heading\n\nContent to be cleared."
    doc = MarkdownDocument(content=content, metadata=metadata)

    doc.clear_content()
    assert doc.content == ""
    assert doc.metadata == metadata


def test_clear_frontmatter():
    metadata = {"title": "Clear Front Matter Test", "tags": ["clear", "test"]}
    content = "# Heading\n\nContent remains."
    doc = MarkdownDocument(content=content, metadata=metadata)

    doc.clear_frontmatter()
    assert doc.metadata == {}
    assert doc.content == content


def test_content_property():
    doc = MarkdownDocument()
    assert doc.content == ""

    doc.content = "# New Heading\n\nNew content."
    assert doc.content == "# New Heading\n\nNew content."


def test_metadata_property():
    doc = MarkdownDocument()
    assert doc.metadata == {}

    new_metadata = {"title": "Metadata Property Test", "tags": ["property", "test"]}
    doc.metadata = new_metadata
    assert doc.metadata == new_metadata
