import json
import shutil
import subprocess
import sys
from pathlib import Path

from ures.literature.citation import CitationManager
from ures.literature.citation.bib_cli import REPORT_FILENAME, main
from ures.literature.citation.manager import BibManager

KNOWN_LIBRARY_ACM_ENTRIES = {
    "pytorch_backends_cuda": {
        "type": "misc",
        "severity": "error",
        "missing_required": ["year"],
        "missing_suggested": [],
        "author_count": 1,
    },
    "chenATPAchievingThroughput2025": {
        "type": "article",
        "severity": "ok",
        "missing_required": [],
        "missing_suggested": [],
        "author_count": 6,
    },
    "yang2025gated": {
        "type": "inproceedings",
        "severity": "warning",
        "missing_required": [],
        "missing_suggested": ["publisher", "address", "pages|articleno"],
        "author_count": 3,
    },
    "bishopNeuralNetworksPattern1995": {
        "type": "book",
        "severity": "ok",
        "missing_required": [],
        "missing_suggested": [],
        "author_count": 1,
    },
}

TYPE_REMAPS = {
    "ketkarIntroductionPyTorch2021": "inproceedings",
    "boulwareImprovingProfilingTechniques": "thesis",
}


def _run(monkeypatch, tmp_path, argv):
    monkeypatch.chdir(tmp_path)
    return main(argv)


def _load_report(directory: Path) -> dict:
    report_path = directory / REPORT_FILENAME
    assert report_path.is_file()
    return json.loads(report_path.read_text(encoding="utf-8"))


def _entries_by_key(report):
    files = report["files"]
    assert len(files) == 1
    return {entry["key"]: entry for entry in files[0]["entries"]}


def _severity_counts(report):
    counts = {}
    for entry in report["files"][0]["entries"]:
        severity = entry["severity"]
        counts[severity] = counts.get(severity, 0) + 1
    return counts


class TestCorpusCliCheck:
    def test_check_writes_structured_report_and_does_not_mutate_source(
        self, acm_library_check, references_inventory
    ):
        assert acm_library_check["code"] == 1
        report = acm_library_check["report"]
        assert set(report) == {"profile", "style", "files"}
        assert report["profile"] == "library"
        assert report["style"] == "acm"

        file_report = report["files"][0]
        assert set(file_report) >= {"path", "drift", "parse_failures", "entries"}
        assert file_report["drift"] is True
        assert Path(file_report["path"]).name == "references.bib"

        parsed = file_report["entries"]
        failures = file_report["parse_failures"]
        assert parsed
        assert failures
        assert len(parsed) + len(failures) == len(references_inventory)
        assert acm_library_check["bib_path"].read_bytes() == acm_library_check[
            "original_bytes"
        ]

    def test_report_entries_have_stable_fields_for_known_keys(self, acm_library_check):
        by_key = _entries_by_key(acm_library_check["report"])
        for key, expected in KNOWN_LIBRARY_ACM_ENTRIES.items():
            assert key in by_key, key
            actual = by_key[key]
            assert actual["key"] == key
            assert actual["type"] == expected["type"]
            assert actual["severity"] == expected["severity"]
            assert actual["missing_required"] == expected["missing_required"]
            assert actual["missing_suggested"] == expected["missing_suggested"]
            assert actual["author_count"] == expected["author_count"]

    def test_parse_failures_are_duplicate_fields_from_the_corpus(
        self, acm_library_check, references_inventory
    ):
        source_keys = {key for _entry_type, key in references_inventory}
        failures = acm_library_check["report"]["files"][0]["parse_failures"]
        assert {item["reason"] for item in failures} == {"DuplicateFieldKeyBlock"}
        assert {item["key"] for item in failures} <= source_keys
        assert "liHowEvaluateSolutions2020" in {item["key"] for item in failures}

    def test_type_normalization_for_incollection_and_phdthesis(self, acm_library_check):
        by_key = _entries_by_key(acm_library_check["report"])
        for key, expected_type in TYPE_REMAPS.items():
            assert by_key[key]["type"] == expected_type


class TestCorpusCliProfilesAndStyles:
    def test_stricter_profiles_raise_more_required_field_errors(
        self, monkeypatch, tmp_path, corpus_bib
    ):
        counts = {}
        for profile in ("library", "submission", "camera-ready"):
            work = tmp_path / profile
            work.mkdir()
            bib = work / "references.bib"
            shutil.copy2(corpus_bib, bib)
            code = _run(monkeypatch, work, ["check", str(bib), "--profile", profile])
            assert code == 1
            report = _load_report(work)
            assert report["profile"] == profile
            assert report["style"] == "acm"
            counts[profile] = _severity_counts(report)

        assert counts["library"]["error"] < counts["submission"]["error"]
        assert counts["submission"]["error"] < counts["camera-ready"]["error"]
        assert counts["library"]["ok"] == counts["camera-ready"]["ok"]
        assert "warning" not in counts["camera-ready"]

    def test_ieee_init_then_check_records_ieee_style(
        self, monkeypatch, tmp_path, corpus_bib
    ):
        code = _run(monkeypatch, tmp_path, ["init", "--style", "ieee"])
        assert code == 0
        style_path = tmp_path / "bibstyle.json"
        assert json.loads(style_path.read_text(encoding="utf-8"))["name"] == "ieee"

        check_code = _run(monkeypatch, tmp_path, ["check", str(corpus_bib)])
        assert check_code == 1
        report = _load_report(tmp_path)
        assert report["style"] == "ieee"
        by_key = _entries_by_key(report)
        assert by_key["chenATPAchievingThroughput2025"]["severity"] == "ok"
        assert by_key["pytorch_backends_cuda"]["missing_required"] == ["year"]


class TestCorpusCliFormat:
    def test_format_rewrites_file_and_keeps_entries_loadable(
        self, monkeypatch, tmp_path, corpus_bib, references_bib_path
    ):
        original = corpus_bib.read_bytes()
        code = _run(monkeypatch, tmp_path, ["format", str(corpus_bib)])
        assert code == 1
        assert corpus_bib.read_bytes() != original
        assert references_bib_path.read_bytes() == original

        report = _load_report(tmp_path)
        file_report = report["files"][0]
        assert file_report["drift"] is True
        formatted_keys = {entry["key"] for entry in file_report["entries"]}
        assert "chenATPAchievingThroughput2025" in formatted_keys
        assert "yang2025gated" in formatted_keys

        reloaded = BibManager(bib_file_path=corpus_bib, bibliography_style="acm")
        assert {entry.key for entry in reloaded.bibliography_entity} == formatted_keys
        text = corpus_bib.read_text(encoding="utf-8")
        assert "chenATPAchievingThroughput2025" in text
        assert "ATP" in text


class TestCorpusCliSubprocess:
    def test_ures_bib_console_script_check_writes_report(
        self, tmp_path, corpus_bib
    ):
        script = Path(sys.executable).parent / "ures-bib"
        if not script.is_file():
            raise AssertionError(f"ures-bib console script missing: {script}")

        proc = subprocess.run(
            [str(script), "check", str(corpus_bib), "--profile", "library"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 1
        assert "error:" in proc.stderr
        assert "warning:" in proc.stderr
        report = _load_report(tmp_path)
        assert report["profile"] == "library"
        assert report["style"] == "acm"
        assert (tmp_path / REPORT_FILENAME).is_file()
        by_key = _entries_by_key(report)
        assert by_key["pytorch_backends_cuda"]["severity"] == "error"
        assert by_key["chenATPAchievingThroughput2025"]["severity"] == "ok"


class TestCorpusBibManager:
    def test_load_export_round_trip_preserves_known_entries(
        self, tmp_path, references_bib_path, references_inventory
    ):
        manager = BibManager(
            bib_file_path=references_bib_path, bibliography_style="acm"
        )
        parsed_keys = {entry.key for entry in manager.bibliography_entity}
        assert (
            len(parsed_keys) + len(manager.failed_blocks)
            == len(references_inventory)
        )
        assert "chenATPAchievingThroughput2025" in parsed_keys
        article = manager.get_entity("chenATPAchievingThroughput2025")
        assert article is not None
        assert article.entry_type == "article"
        assert "ATP" in article["title"]

        exported = tmp_path / "exported.bib"
        manager.export_to_file(str(exported), manager.bibliograph_library)
        text = exported.read_text(encoding="utf-8")
        assert "chenATPAchievingThroughput2025" in text
        assert "10.1145/3701996" in text

        reloaded = BibManager(bib_file_path=exported, bibliography_style="acm")
        assert {entry.key for entry in reloaded.bibliography_entity} == parsed_keys
        assert reloaded.get_entity("yang2025gated") is not None


class TestCorpusCitationManager:
    def test_import_from_tex_and_save_cited_subset(
        self, tmp_path, references_bib_path
    ):
        tex = tmp_path / "paper.tex"
        tex.write_text(
            "Related work includes "
            r"\cite{chenATPAchievingThroughput2025,yang2025gated} "
            r"and docs \cite{pytorch_backends_cuda} "
            r"plus a missing key \cite{not-in-this-bib}."
            "\n",
            encoding="utf-8",
        )
        manager = CitationManager(
            bibliography_files=str(references_bib_path),
            bibliography_style="acm",
        )
        imported = manager.import_citations(files=[tex])
        assert set(imported) == {
            "chenATPAchievingThroughput2025",
            "yang2025gated",
            "pytorch_backends_cuda",
            "not-in-this-bib",
        }
        assert imported["chenATPAchievingThroughput2025"].bibliography is not None
        assert imported["yang2025gated"].bibliography is not None
        assert imported["pytorch_backends_cuda"].bibliography is not None
        assert imported["not-in-this-bib"].bibliography is None

        cited_bib = tmp_path / "cited.bib"
        manager.save_bibliography(str(cited_bib))
        text = cited_bib.read_text(encoding="utf-8")
        assert "chenATPAchievingThroughput2025" in text
        assert "yang2025gated" in text
        assert "pytorch_backends_cuda" in text
        assert "not-in-this-bib" not in text
        assert "bishopNeuralNetworksPattern1995" not in text

        saved = BibManager(bib_file_path=cited_bib, bibliography_style="acm")
        assert {entry.key for entry in saved.bibliography_entity} == {
            "chenATPAchievingThroughput2025",
            "yang2025gated",
            "pytorch_backends_cuda",
        }
