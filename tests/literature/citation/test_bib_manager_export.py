import json

from ures.literature.citation.manager import BibManager
from ures.literature.citation.rules import BibRuleRegister


MANY_AUTHORS = (
    "Alice A and Bob B and Carol C and Dan D and Eve E and Frank F"
)


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
