import json

import pytest
from bibtexparser.model import Entry, Field

from ures.literature.citation.manager import BibManager
from ures.literature.citation.middlewares import RuleBasedValidationMiddleware
from ures.literature.citation.rules import BibRuleRegister, StyleConfigError
from ures.literature.citation.rules.basic import DefaultRules
from ures.literature.citation.rules.style_config import (
    apply_profile,
    resolve_style_document,
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _field_list(entry, key):
    field = entry.get(key, None)
    if field is None:
        return []
    return list(field.value or [])


def _load_article(tmp_path, extra_fields, *, style="acm", profile="library"):
    body = [
        "@article{Smith2024,",
        "  author = {Smith, John},",
        "  title = {A Title},",
        "  year = {2024},",
        "  journal = {Some Journal},",
    ]
    body.extend(f"  {line}" for line in extra_fields)
    body.append("}")
    bib_path = tmp_path / "refs.bib"
    bib_path.write_text("\n".join(body) + "\n", encoding="utf-8")
    manager = BibManager(
        bib_file_path=bib_path, bibliography_style=style, profile=profile
    )
    return manager.bibliography_entity[0]


class TestStyleLoad:
    def test_packaged_acm_loads_library_required_without_month(self):
        style = resolve_style_document(style_name="acm")
        assert style["name"] == "acm"
        assert style["max_authors"] == 0
        assert style["proceedings_style"] == "proceedings"
        article = style["entry_types"]["article"]
        assert article["required"] == ["author", "title", "year", "journal"]
        assert "month" not in article["required"]
        assert "pages|articleno" in article["suggested"]
        assert "doi|url" in article["suggested"]

        register = BibRuleRegister.from_json_style(style="acm", profile="library")
        rule = register.get_rule("article")
        assert rule.standard_name == "acm"
        assert "month" not in rule.required_fields
        assert "journal" in rule.required_fields
        book = register.get_rule("book")
        assert "publisher" in book.required_fields
        assert "address" in book.required_fields

    def test_packaged_ieee_loads_distinct_baseline(self):
        style = resolve_style_document(style_name="ieee")
        assert style["name"] == "ieee"
        assert style["proceedings_style"] == "proc"
        article = style["entry_types"]["article"]
        assert article["required"] == ["author", "title", "year", "journal"]
        assert article["suggested"] == ["volume", "pages", "url"]
        book = style["entry_types"]["book"]
        assert book["required"] == ["author", "title", "year", "publisher"]
        assert "address" not in book["required"]
        assert "address" in book["suggested"]

        register = BibRuleRegister.from_json_style(style="ieee")
        assert register.style_name == "ieee"
        assert "address" not in register.get_rule("book").required_fields

    def test_unknown_style_raises(self):
        with pytest.raises(StyleConfigError, match="unknown style 'foo'"):
            resolve_style_document(style_name="foo")

    def test_bib_rule_register_no_args_keeps_python_default_rules(self):
        register = BibRuleRegister()
        rules = register.get_rules()
        assert len(rules) == len(DefaultRules)
        assert all(rule.standard_name == "default" for rule in rules)
        article = register.get_rule("article")
        assert "month" in article.required_fields

    def test_bib_manager_default_and_acm_use_json_not_python_month_list(self, tmp_path):
        entry = _load_article(tmp_path, extra_fields=[], style="default")
        missing = _field_list(entry, "missing_fields")
        assert "month" not in missing
        assert "journal" not in missing
        suggested = _field_list(entry, "missing_suggested")
        assert suggested


class TestOverlay:
    def test_overlay_add_and_remove(self, tmp_path):
        overlay = {
            "extends": "ieee",
            "max_authors": 8,
            "proceedings_style": "proc",
            "entry_types": {
                "inproceedings": {
                    "required_add": ["pages"],
                    "suggested_add": ["doi"],
                    "suggested_remove": ["address"],
                }
            },
        }
        path = _write_json(tmp_path / "bibstyle.json", overlay)
        merged = resolve_style_document(local_path=path)
        assert merged["name"] == "ieee"
        assert merged["max_authors"] == 8
        inproc = merged["entry_types"]["inproceedings"]
        assert inproc["required"][-1] == "pages"
        assert "pages" in inproc["required"]
        assert "doi" in inproc["suggested"]
        assert "address" not in inproc["suggested"]
        article = merged["entry_types"]["article"]
        assert article["required"] == ["author", "title", "year", "journal"]

    def test_overlay_without_extends_defaults_to_acm(self, tmp_path):
        overlay = {
            "entry_types": {
                "article": {
                    "suggested_remove": ["volume"],
                    "suggested_add": ["issn"],
                }
            }
        }
        merged = resolve_style_document(
            local_path=_write_json(tmp_path / "bibstyle.json", overlay)
        )
        assert merged["name"] == "acm"
        suggested = merged["entry_types"]["article"]["suggested"]
        assert "volume" not in suggested
        assert "issn" in suggested

    def test_unknown_overlay_type_starts_from_core_required(self, tmp_path):
        overlay = {
            "extends": "acm",
            "entry_types": {
                "customtype": {
                    "required_add": ["url"],
                    "suggested_add": ["note"],
                }
            },
        }
        merged = resolve_style_document(
            local_path=_write_json(tmp_path / "overlay.json", overlay)
        )
        spec = merged["entry_types"]["customtype"]
        assert spec["required"] == ["author", "title", "year", "url"]
        assert spec["suggested"] == ["note"]

    def test_mixed_extends_and_complete_lists_is_config_error(self, tmp_path):
        mixed = {
            "extends": "acm",
            "entry_types": {
                "article": {
                    "required": ["author", "title", "year", "journal"],
                    "required_add": ["issn"],
                }
            },
        }
        path = _write_json(tmp_path / "bibstyle.json", mixed)
        with pytest.raises(StyleConfigError, match="cannot mix extends"):
            resolve_style_document(local_path=path)


class TestProtectedRequiredRemove:
    @pytest.mark.parametrize("field_name", ["author", "title", "year"])
    def test_required_remove_of_core_fields_raises(self, tmp_path, field_name):
        overlay = {
            "extends": "acm",
            "entry_types": {
                "article": {"required_remove": [field_name]},
            },
        }
        path = _write_json(tmp_path / "bibstyle.json", overlay)
        with pytest.raises(
            StyleConfigError,
            match=rf"cannot remove required field {field_name}",
        ) as exc_info:
            resolve_style_document(local_path=path)
        assert isinstance(exc_info.value, StyleConfigError)

    def test_required_remove_of_non_core_field_is_allowed(self, tmp_path):
        overlay = {
            "extends": "acm",
            "entry_types": {
                "article": {"required_remove": ["journal"]},
            },
        }
        merged = resolve_style_document(
            local_path=_write_json(tmp_path / "overlay.json", overlay)
        )
        assert merged["entry_types"]["article"]["required"] == [
            "author",
            "title",
            "year",
        ]


class TestProfilePromotion:
    def test_library_keeps_suggested_as_warnings(self):
        style = apply_profile(resolve_style_document(style_name="acm"), "library")
        article = style["entry_types"]["article"]
        assert article["required"] == ["author", "title", "year", "journal"]
        assert "doi|url" in article["suggested"]
        assert "doi|url" not in article["required"]

    def test_acm_submission_promotes_doi_or_url_only(self):
        style = apply_profile(resolve_style_document(style_name="acm"), "submission")
        article = style["entry_types"]["article"]
        assert "doi|url" in article["required"]
        assert "doi|url" not in article["suggested"]
        assert "volume" in article["suggested"]
        assert "volume" not in article["required"]

    def test_ieee_submission_promotes_url_not_doi(self):
        style = apply_profile(resolve_style_document(style_name="ieee"), "submission")
        article = style["entry_types"]["article"]
        assert "url" in article["required"]
        assert "doi" not in article["required"]
        assert "doi|url" not in article["required"]
        assert "volume" in article["suggested"]

    def test_camera_ready_promotes_all_suggested(self):
        acm = apply_profile(resolve_style_document(style_name="acm"), "camera-ready")
        acm_article = acm["entry_types"]["article"]
        assert acm_article["suggested"] == []
        for field_name in ["volume", "number", "pages|articleno", "doi|url"]:
            assert field_name in acm_article["required"]

        ieee = apply_profile(resolve_style_document(style_name="ieee"), "camera-ready")
        ieee_article = ieee["entry_types"]["article"]
        assert ieee_article["suggested"] == []
        assert "url" in ieee_article["required"]
        assert "doi" not in ieee_article["required"]
        assert "doi|url" not in ieee_article["required"]

    def test_unknown_profile_raises(self):
        with pytest.raises(StyleConfigError, match="unknown profile 'strict'"):
            apply_profile(resolve_style_document(style_name="acm"), "strict")

    def test_loaded_entries_follow_profile_severity(self, tmp_path):
        library = _load_article(tmp_path, extra_fields=[], profile="library")
        assert _field_list(library, "missing_fields") == []
        assert _field_list(library, "missing_suggested")
        assert library.get("is_valid").value is True

        submission = _load_article(tmp_path, extra_fields=[], profile="submission")
        assert "doi|url" in _field_list(submission, "missing_fields")
        assert "volume" in _field_list(submission, "missing_suggested")
        assert submission.get("is_valid").value is False

        camera = _load_article(tmp_path, extra_fields=[], profile="camera-ready")
        missing = _field_list(camera, "missing_fields")
        assert "doi|url" in missing
        assert "volume" in missing
        assert _field_list(camera, "missing_suggested") == []
        assert camera.get("is_valid").value is False


class TestIdentifierAndPagesEquivalence:
    def _validate(self, tmp_path, extra_fields, *, style="acm", profile="camera-ready"):
        return _load_article(
            tmp_path, extra_fields, style=style, profile=profile
        )

    def test_acm_url_without_doi_satisfies_identifier(self, tmp_path):
        extra = [
            "volume = {1},",
            "number = {2},",
            "pages = {1--10},",
            "url = {https://example.com/paper},",
        ]
        entry = self._validate(tmp_path, extra)
        assert "doi|url" not in _field_list(entry, "missing_fields")
        assert entry.get("is_valid").value is True

    def test_acm_doi_without_url_satisfies_identifier(self, tmp_path):
        extra = [
            "volume = {1},",
            "number = {2},",
            "pages = {1--10},",
            "doi = {10.1145/example},",
        ]
        entry = self._validate(tmp_path, extra)
        assert "doi|url" not in _field_list(entry, "missing_fields")
        assert entry.get("is_valid").value is True

    def test_acm_articleno_without_pages_satisfies_pages_group(self, tmp_path):
        extra = [
            "volume = {1},",
            "number = {2},",
            "articleno = {15},",
            "doi = {10.1145/example},",
        ]
        entry = self._validate(tmp_path, extra)
        assert "pages|articleno" not in _field_list(entry, "missing_fields")
        assert entry.get("is_valid").value is True

    def test_acm_pages_without_articleno_satisfies_pages_group(self, tmp_path):
        extra = [
            "volume = {1},",
            "number = {2},",
            "pages = {10--20},",
            "url = {https://example.com/paper},",
        ]
        entry = self._validate(tmp_path, extra)
        assert "pages|articleno" not in _field_list(entry, "missing_fields")

    def test_acm_missing_both_identifier_fields_is_required_error(self, tmp_path):
        extra = [
            "volume = {1},",
            "number = {2},",
            "pages = {1--10},",
        ]
        entry = self._validate(tmp_path, extra)
        assert "doi|url" in _field_list(entry, "missing_fields")
        assert entry.get("is_valid").value is False

    def test_ieee_doi_does_not_satisfy_url_requirement(self, tmp_path):
        extra = [
            "volume = {1},",
            "pages = {1--10},",
            "doi = {10.1109/example},",
        ]
        entry = self._validate(
            tmp_path, extra, style="ieee", profile="camera-ready"
        )
        assert "url" in _field_list(entry, "missing_fields")
        assert "doi" not in _field_list(entry, "missing_fields")

    def test_ieee_url_without_doi_is_valid_camera_ready(self, tmp_path):
        extra = [
            "volume = {1},",
            "pages = {1--10},",
            "url = {https://ieeexplore.ieee.org/document/1},",
        ]
        entry = self._validate(
            tmp_path, extra, style="ieee", profile="camera-ready"
        )
        assert _field_list(entry, "missing_fields") == []
        assert entry.get("is_valid").value is True

    def test_field_group_on_register_rules_directly(self):
        register = BibRuleRegister.from_json_style(style="acm", profile="camera-ready")
        middleware = RuleBasedValidationMiddleware(rule_register=register)
        entry = Entry(
            key="OnlyUrl",
            entry_type="article",
            fields=[
                Field(key="author", value="Smith, John"),
                Field(key="title", value="A Title"),
                Field(key="year", value="2024"),
                Field(key="journal", value="Some Journal"),
                Field(key="volume", value="1"),
                Field(key="number", value="2"),
                Field(key="articleno", value="9"),
                Field(key="url", value="https://example.com"),
            ],
        )
        result = middleware.transform_entry(entry)
        assert result.get("is_valid").value is True
        assert _field_list(result, "missing_fields") == []
