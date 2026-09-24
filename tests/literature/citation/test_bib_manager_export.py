import json

from ures.literature.citation.manager import BibManager
from ures.literature.citation.rules import BibRuleRegister


MANY_AUTHORS = "Alice A and Bob B and Carol C and Dan D and Eve E and Frank F"


def _article_bib(authors=MANY_AUTHORS, extra=""):
    extra_block = f"  {extra}\n" if extra else ""
    return (
        "@article{Smith2024,\n"
        f"  author = {{{authors}}},\n"
        "  title = {A Title},\n"
        "  year = {2024},\n"
        "  journal = {Some Journal},\n"
        f"{extra_block}"
        "}\n"
    )


class TestExportKeepsExtraFields:
    def test_acm_export_keeps_fields_outside_required_and_suggested(self, tmp_path):
        src = tmp_path / "in.bib"
        src.write_text(
            _article_bib(
                authors="Smith, John",
                extra="note = {keep this note},\n  abstract = {keep this abstract},",
            ),
            encoding="utf-8",
        )
        manager = BibManager(bib_file_path=src, bibliography_style="acm")
        out = tmp_path / "out.bib"
        manager.export_to_file(str(out), manager.bibliograph_library)
        text = out.read_text(encoding="utf-8")
        assert "keep this note" in text
        assert "keep this abstract" in text
        assert "note" in text
        assert "abstract" in text

    def test_ieee_export_keeps_doi_as_extra_field(self, tmp_path):
        src = tmp_path / "in.bib"
        src.write_text(
            _article_bib(
                authors="Smith, John",
                extra="doi = {10.1109/example},\n  url = {https://example.com},",
            ),
            encoding="utf-8",
        )
        manager = BibManager(bib_file_path=src, bibliography_style="ieee")
        out = tmp_path / "out.bib"
        manager.export_to_file(str(out), manager.bibliograph_library)
        text = out.read_text(encoding="utf-8")
        assert "10.1109/example" in text
        assert "doi" in text.lower()


class TestExportDropsRemovedFields:
    def test_overlay_strips_removed_fields_and_keeps_others(self, tmp_path):
        overlay = tmp_path / "bibstyle.json"
        overlay.write_text(
            json.dumps(
                {
                    "extends": "acm",
                    "entry_types": {
                        "inproceedings": {
                            "suggested_remove": ["location|city"],
                        },
                        "book": {"required_remove": ["address"]},
                    },
                }
            ),
            encoding="utf-8",
        )
        src = tmp_path / "in.bib"
        src.write_text(
            "@inproceedings{Ansel2024,\n"
            "  author = {Ansel, Jason},\n"
            "  title = {PyTorch 2},\n"
            "  year = {2024},\n"
            "  booktitle = {Proceedings of ASPLOS},\n"
            "  location = {La Jolla, CA, USA},\n"
            "  city = {La Jolla},\n"
            "  address = {New York},\n"
            "  note = {keep this note},\n"
            "}\n"
            "@book{Knuth1997,\n"
            "  author = {Knuth, Donald},\n"
            "  title = {The Art of Computer Programming},\n"
            "  year = {1997},\n"
            "  publisher = {Addison-Wesley},\n"
            "  address = {Reading, Mass.},\n"
            "  isbn = {0201896834},\n"
            "}\n",
            encoding="utf-8",
        )
        rules = BibRuleRegister.from_json_style(local_path=overlay)
        inproc = rules.get_rule("inproceedings")
        assert inproc.forbidden_fields == ["location", "city"]
        assert rules.get_rule("book").forbidden_fields == ["address"]
        manager = BibManager(bib_file_path=src, rules=rules)
        out = tmp_path / "out.bib"
        manager.export_to_file(str(out), manager.bibliograph_library)
        text = out.read_text(encoding="utf-8")
        assert "La Jolla, CA, USA" not in text
        assert "city = {La Jolla}" not in text
        assert "address = {New York}" in text
        assert "keep this note" in text
        assert "Reading, Mass." not in text
        assert "0201896834" in text


class TestMaxAuthorsExport:
    def test_max_authors_zero_does_not_truncate(self, tmp_path):
        src = tmp_path / "in.bib"
        src.write_text(_article_bib(), encoding="utf-8")
        manager = BibManager(bib_file_path=src, bibliography_style="acm")
        assert manager.rules.max_authors == 0
        out = tmp_path / "out.bib"
        manager.export_to_file(str(out), manager.bibliograph_library)
        text = out.read_text(encoding="utf-8").lower()
        for name in ["alice", "bob", "carol", "dan", "eve", "frank"]:
            assert name in text
        assert "others" not in text

    def test_positive_max_authors_truncates_on_export(self, tmp_path):
        overlay = tmp_path / "bibstyle.json"
        overlay.write_text(
            json.dumps({"extends": "acm", "max_authors": 2}),
            encoding="utf-8",
        )
        src = tmp_path / "in.bib"
        src.write_text(_article_bib(), encoding="utf-8")
        rules = BibRuleRegister.from_json_style(local_path=overlay)
        manager = BibManager(bib_file_path=src, rules=rules)
        assert manager.rules.max_authors == 2
        out = tmp_path / "out.bib"
        manager.export_to_file(str(out), manager.bibliograph_library)
        text = out.read_text(encoding="utf-8").lower()
        assert "alice" in text
        assert "bob" in text
        assert "frank" not in text
        assert "others" in text
