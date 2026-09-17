import json

from ures.literature.citation.bib_cli import REPORT_FILENAME, main
from ures.literature.citation.rules import BibRuleRegister
from ures.literature.citation.rules.style_config import resolve_style_document


def _run(monkeypatch, tmp_path, capsys, argv):
    monkeypatch.chdir(tmp_path)
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _write_bib(path, contents):
    path.write_text(contents, encoding="utf-8")
    return path


COMPLETE_ARTICLE = """\
@article{Smith2024,
  author = {Smith, John},
  title = {A Title},
  year = {2024},
  journal = {Some Journal},
}
"""

UNNORMALIZED_ARTICLE = """\
@article{Smith2024,
  author = {Smith, John},
  title = {A Title},
  year = {2024},
  journaltitle = {Some Journal},
  pages = {1-10},
}
"""


class TestInit:
    def test_init_writes_json_that_reloads_as_that_style(
        self, monkeypatch, tmp_path, capsys
    ):
        code, _out, err = _run(
            monkeypatch, tmp_path, capsys, ["init", "--style", "ieee"]
        )
        assert code == 0
        dest = tmp_path / "bibstyle.json"
        assert dest.is_file()
        written = json.loads(dest.read_text(encoding="utf-8"))
        assert written["name"] == "ieee"
        assert written["proceedings_style"] == "proc"
        assert "extends" not in written

        resolved = resolve_style_document(local_path=dest)
        assert resolved["name"] == "ieee"
        assert resolved["entry_types"]["book"]["required"] == [
            "author",
            "title",
            "year",
            "publisher",
        ]

        register = BibRuleRegister.from_json_style(load_cwd=True)
        assert register.style_name == "ieee"
        assert "address" not in register.get_rule("book").required_fields
        assert "bibstyle.json" in err or str(dest) in err

    def test_init_default_style_is_acm(self, monkeypatch, tmp_path, capsys):
        code, _out, _err = _run(monkeypatch, tmp_path, capsys, ["init"])
        assert code == 0
        written = json.loads((tmp_path / "bibstyle.json").read_text(encoding="utf-8"))
        assert written["name"] == "acm"
        resolved = resolve_style_document(local_path=tmp_path / "bibstyle.json")
        assert resolved["name"] == "acm"
        assert "address" in resolved["entry_types"]["book"]["required"]

    def test_init_exits_2_when_bibstyle_exists(self, monkeypatch, tmp_path, capsys):
        existing = tmp_path / "bibstyle.json"
        existing.write_text('{"name": "acm"}\n', encoding="utf-8")
        code, _out, err = _run(monkeypatch, tmp_path, capsys, ["init"])
        assert code == 2
        assert "error:" in err
        assert "already exists" in err
        assert existing.read_text(encoding="utf-8") == '{"name": "acm"}\n'


class TestCheckAndFormat:
    def test_check_unnormalized_file_exits_1_and_prints_error(
        self, monkeypatch, tmp_path, capsys
    ):
        bib = _write_bib(tmp_path / "refs.bib", UNNORMALIZED_ARTICLE)
        code, _out, err = _run(monkeypatch, tmp_path, capsys, ["check", str(bib)])
        assert code == 1
        assert "error:" in err
        lines = [line for line in err.splitlines() if line.startswith("error:")]
        assert lines
        assert any("formatting differs" in line for line in lines) or any(
            line.startswith("error:") for line in err.splitlines()
        )
        assert bib.read_text(encoding="utf-8") == UNNORMALIZED_ARTICLE
        assert (tmp_path / REPORT_FILENAME).is_file()

    def test_format_then_check_on_complete_entry_can_exit_0(
        self, monkeypatch, tmp_path, capsys
    ):
        bib = _write_bib(tmp_path / "refs.bib", COMPLETE_ARTICLE)
        format_code, _out, format_err = _run(
            monkeypatch, tmp_path, capsys, ["format", str(bib)]
        )
        assert format_code == 0, format_err
        assert bib.read_text(encoding="utf-8") != ""
        assert (tmp_path / REPORT_FILENAME).is_file()

        check_code, _out, check_err = _run(
            monkeypatch, tmp_path, capsys, ["check", str(bib)]
        )
        assert check_code == 0, check_err
        assert not any(
            line.startswith("error:") for line in check_err.splitlines()
        )

    def test_unknown_profile_prints_error_and_exits_2(
        self, monkeypatch, tmp_path, capsys
    ):
        code, _out, err = _run(
            monkeypatch,
            tmp_path,
            capsys,
            ["check", "refs.bib", "--profile", "not-a-profile"],
        )
        assert code == 2
        assert "error:" in err
        assert "unknown profile" in err
        first_error = next(
            line for line in err.splitlines() if line.startswith("error:")
        )
        assert first_error.startswith("error:")

    def test_required_remove_of_author_exits_2(self, monkeypatch, tmp_path, capsys):
        (tmp_path / "bibstyle.json").write_text(
            json.dumps(
                {
                    "extends": "acm",
                    "entry_types": {"article": {"required_remove": ["author"]}},
                }
            ),
            encoding="utf-8",
        )
        bib = _write_bib(tmp_path / "refs.bib", COMPLETE_ARTICLE)
        code, _out, err = _run(monkeypatch, tmp_path, capsys, ["check", str(bib)])
        assert code == 2
        assert err.startswith("error:") or "error:" in err
        assert "cannot remove required field author" in err
