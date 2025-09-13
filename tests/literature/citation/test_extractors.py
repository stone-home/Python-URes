import pytest
import tempfile
import bibtexparser
from pathlib import Path
from unittest.mock import patch, mock_open
from ures.literature.citation.extractors import (
    CitationSource,
    CitationInfo,
    AbcCitationExtractor,
    TexCitationExtractor,
    BBLCitationExtractor,
)


class TestCitationSource:
    """Test class for CitationSource data structure."""

    def test_citation_source_creation(self):
        """Test creating a CitationSource instance."""
        source = CitationSource(
            source_file="test.tex", line_number=10, source_type="tex"
        )
        assert source.source_file == "test.tex"
        assert source.line_number == 10
        assert source.source_type == "tex"

    def test_citation_source_immutable(self):
        """Test that CitationSource is immutable (frozen dataclass)."""
        source = CitationSource(
            source_file="test.tex", line_number=10, source_type="tex"
        )
        with pytest.raises(AttributeError):
            source.source_file = "modified.tex"

    def test_citation_source_equality(self):
        """Test equality comparison between CitationSource instances."""
        source1 = CitationSource("test.tex", 10, "tex")
        source2 = CitationSource("test.tex", 10, "tex")
        source3 = CitationSource("test.tex", 11, "tex")

        assert source1 == source2
        assert source1 != source3


class TestCitationInfo:
    """Test class for CitationInfo data structure."""

    def test_citation_info_creation_with_defaults(self):
        """Test creating CitationInfo with default values."""
        info = CitationInfo(key="test_key")
        assert info.key == "test_key"
        assert info.sources == []
        assert info.bibliography is None

    def test_citation_info_creation_with_sources(self):
        """Test creating CitationInfo with sources."""
        source = CitationSource("test.tex", 5, "tex")
        info = CitationInfo(key="test_key", sources=[source])

        assert info.key == "test_key"
        assert len(info.sources) == 1
        assert info.sources[0] == source

    def test_citation_info_repr_without_bibliography(self):
        """Test __repr__ method when bibliography is None."""
        info = CitationInfo(key="test_key")
        repr_str = repr(info)
        assert "[False]Cite Key: test_key" == repr_str

    def test_citation_info_repr_with_valid_bibliography(self):
        """Test __repr__ method with valid bibliography."""
        # Create a mock bibliography entry
        field = bibtexparser.model.Field(key="is_valid", value=True)
        mock_entry = bibtexparser.model.Entry(
            entry_type="article", key="test_key", fields=[field]
        )

        info = CitationInfo(key="test_key", bibliography=mock_entry)
        repr_str = repr(info)
        assert "[True]Cite Key: test_key" == repr_str

    def test_citation_info_repr_with_invalid_bibliography(self):
        """Test __repr__ method with invalid bibliography."""
        field = bibtexparser.model.Field(key="is_valid", value=False)
        mock_entry = bibtexparser.model.Entry(
            entry_type="article", key="test_key", fields=[field]
        )

        info = CitationInfo(key="test_key", bibliography=mock_entry)
        repr_str = repr(info)
        assert "[False]Cite Key: test_key" == repr_str


class TestAbcCitationExtractor:
    """Test class for abstract base class AbcCitationExtractor."""

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that abstract base class cannot be instantiated."""
        with pytest.raises(TypeError):
            AbcCitationExtractor()

    def test_subclass_must_implement_extract_citations(self):
        """Test that subclasses must implement extract_citations method."""

        class IncompleteExtractor(AbcCitationExtractor):
            pass

        with pytest.raises(TypeError):
            IncompleteExtractor()


class TestTexCitationExtractor:
    """Test class for TexCitationExtractor."""

    def setup_method(self):
        """Setup method called before each test method."""
        self.extractor = TexCitationExtractor()

    def test_extract_citations_simple_cite(self):
        """Test extraction of simple \\cite{} citations."""
        tex_content = """
        This is a test document.
        Here is a citation \\cite{author2023}.
        Another line without citation.
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as f:
            f.write(tex_content)
            f.flush()

            citations = self.extractor.extract_citations(f.name)

        assert len(citations) == 1
        assert citations[0].key == "author2023"
        assert len(citations[0].sources) == 1
        assert citations[0].sources[0].line_number == 3
        assert citations[0].sources[0].source_type == "tex"

    def test_extract_citations_multiple_keys_single_cite(self):
        """Test extraction of multiple keys in single citation command."""
        tex_content = """\\cite{author2023, smith2024, jones2022}"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as f:
            f.write(tex_content)
            f.flush()

            citations = self.extractor.extract_citations(f.name)

        assert len(citations) == 3
        citation_keys = {c.key for c in citations}
        assert citation_keys == {"author2023", "smith2024", "jones2022"}

    def test_extract_citations_with_optional_arguments(self):
        """Test extraction of citations with optional arguments."""
        tex_content = """\\cite[p. 123]{author2023}"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as f:
            f.write(tex_content)
            f.flush()

            citations = self.extractor.extract_citations(f.name)

        assert len(citations) == 1
        assert citations[0].key == "author2023"

    def test_extract_citations_various_citation_commands(self):
        """Test extraction of various citation command types."""
        tex_content = """
        \\cite{cite_key}
        \\citep{citep_key}
        \\citet{citet_key}
        \\citeauthor{citeauthor_key}
        \\citeyear{citeyear_key}
        \\citealp{citealp_key}
        \\citealt{citealt_key}
        \\parencite{parencite_key}
        \\textcite{textcite_key}
        \\nocite{nocite_key}
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as f:
            f.write(tex_content)
            f.flush()

            citations = self.extractor.extract_citations(f.name)

        expected_keys = {
            "cite_key",
            "citep_key",
            "citet_key",
            "citeauthor_key",
            "citeyear_key",
            "citealp_key",
            "citealt_key",
            "parencite_key",
            "textcite_key",
            "nocite_key",
        }
        citation_keys = {c.key for c in citations}
        assert citation_keys == expected_keys

    def test_extract_citations_with_comments(self):
        """Test that citations in comments are ignored."""
        tex_content = """
        \\cite{valid_citation}
        % This is a comment with \\cite{commented_citation}
        Another line with \\cite{another_valid}
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as f:
            f.write(tex_content)
            f.flush()

            citations = self.extractor.extract_citations(f.name)

        citation_keys = {c.key for c in citations}
        assert citation_keys == {"valid_citation", "another_valid"}
        assert "commented_citation" not in citation_keys

    def test_extract_citations_escaped_percent(self):
        """Test that escaped percent signs don't affect citation extraction."""
        tex_content = """
        This is 100\\% certain \\cite{certain_citation}.
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as f:
            f.write(tex_content)
            f.flush()

            citations = self.extractor.extract_citations(f.name)

        assert len(citations) == 1
        assert citations[0].key == "certain_citation"

    def test_extract_citations_duplicate_keys(self):
        """Test handling of duplicate citation keys."""
        tex_content = """
        \\cite{duplicate_key}
        \\citep{duplicate_key}
        \\cite{unique_key}
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as f:
            f.write(tex_content)
            f.flush()

            citations = self.extractor.extract_citations(f.name)

        # Should have 2 unique keys but duplicate_key should have 2 sources
        citation_dict = {c.key: c for c in citations}
        assert len(citation_dict) == 2
        assert len(citation_dict["duplicate_key"].sources) == 2
        assert len(citation_dict["unique_key"].sources) == 1

    def test_extract_citations_empty_file(self):
        """Test extraction from empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as f:
            f.write("")
            f.flush()

            citations = self.extractor.extract_citations(f.name)

        assert len(citations) == 0

    def test_extract_citations_no_citations(self):
        """Test extraction from file with no citations."""
        tex_content = """
        This is a document with no citations.
        Just regular text here.
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as f:
            f.write(tex_content)
            f.flush()

            citations = self.extractor.extract_citations(f.name)

        assert len(citations) == 0

    def test_extract_citations_path_object(self):
        """Test that method accepts Path objects."""
        tex_content = """\\cite{path_test}"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as f:
            f.write(tex_content)
            f.flush()

            path_obj = Path(f.name)
            citations = self.extractor.extract_citations(path_obj)

        assert len(citations) == 1
        assert citations[0].key == "path_test"


class TestBBLCitationExtractor:
    """Test class for BBLCitationExtractor."""

    def setup_method(self):
        """Setup method called before each test method."""
        self.extractor = BBLCitationExtractor()

    def test_extract_citations_simple_bibitem(self):
        """Test extraction of simple \\bibitem entries."""
        bbl_content = """
        \\begin{thebibliography}{99}
        \\bibitem{author2023}
        Author, A. (2023). Title of the paper.
        \\end{thebibliography}
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".bbl", delete=False) as f:
            f.write(bbl_content)
            f.flush()

            citations = self.extractor.extract_citations(f.name)

        assert len(citations) == 1
        assert citations[0].key == "author2023"
        assert len(citations[0].sources) == 1
        assert citations[0].sources[0].source_type == "bbl"

    def test_extract_citations_bibitem_with_optional_args(self):
        """Test extraction of \\bibitem entries with optional arguments."""
        bbl_content = """
        \\bibitem[Author, 2023]{author2023}
        Author, A. (2023). Title of the paper.
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".bbl", delete=False) as f:
            f.write(bbl_content)
            f.flush()

            citations = self.extractor.extract_citations(f.name)

        assert len(citations) == 1
        assert citations[0].key == "author2023"

    def test_extract_citations_multiple_bibitems(self):
        """Test extraction of multiple \\bibitem entries."""
        bbl_content = """
        \\begin{thebibliography}{99}
        \\bibitem{author2023}
        Author, A. (2023). First paper.

        \\bibitem[Smith et al., 2024]{smith2024}
        Smith, B. et al. (2024). Second paper.

        \\bibitem{jones2022}
        Jones, C. (2022). Third paper.
        \\end{thebibliography}
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".bbl", delete=False) as f:
            f.write(bbl_content)
            f.flush()

            citations = self.extractor.extract_citations(f.name)

        assert len(citations) == 3
        citation_keys = {c.key for c in citations}
        assert citation_keys == {"author2023", "smith2024", "jones2022"}

    def test_extract_citations_complex_bibitem_content(self):
        """Test extraction from complex \\bibitem with multiline content."""
        bbl_content = """
        \\bibitem{complex_key}
        This is a very long bibliography entry
        that spans multiple lines and contains
        various formatting commands like \\textit{italics}
        and \\textbf{bold} text.
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".bbl", delete=False) as f:
            f.write(bbl_content)
            f.flush()

            citations = self.extractor.extract_citations(f.name)

        assert len(citations) == 1
        assert citations[0].key == "complex_key"

    def test_extract_citations_empty_bbl_file(self):
        """Test extraction from empty BBL file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".bbl", delete=False) as f:
            f.write("")
            f.flush()

            citations = self.extractor.extract_citations(f.name)

        assert len(citations) == 0

    def test_extract_citations_no_bibitems(self):
        """Test extraction from BBL file with no \\bibitem entries."""
        bbl_content = """
        \\begin{thebibliography}{99}
        This is just some text without any bibliography items.
        \\end{thebibliography}
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".bbl", delete=False) as f:
            f.write(bbl_content)
            f.flush()

            citations = self.extractor.extract_citations(f.name)

        assert len(citations) == 0

    def test_extract_citations_file_not_found(self):
        """Test handling of non-existent BBL file."""
        non_existent_file = "/path/to/non/existent/file.bbl"
        citations = self.extractor.extract_citations(non_existent_file)
        assert len(citations) == 0

    def test_extract_citations_path_object(self):
        """Test that method accepts Path objects."""
        bbl_content = """\\bibitem{path_test_key}"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".bbl", delete=False) as f:
            f.write(bbl_content)
            f.flush()

            path_obj = Path(f.name)
            citations = self.extractor.extract_citations(path_obj)

        assert len(citations) == 1
        assert citations[0].key == "path_test_key"

    def test_extract_citations_line_number_tracking(self):
        """Test that line numbers are correctly tracked."""
        bbl_content = """Line 1
Line 2
\\bibitem{key_on_line_3}
Line 4
Line 5
\\bibitem{key_on_line_6}"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".bbl", delete=False) as f:
            f.write(bbl_content)
            f.flush()

            citations = self.extractor.extract_citations(f.name)

        assert len(citations) == 2
        citation_dict = {c.key: c for c in citations}
        assert citation_dict["key_on_line_3"].sources[0].line_number == 3
        assert citation_dict["key_on_line_6"].sources[0].line_number == 6


class TestIntegration:
    """Integration tests for the citation extraction system."""

    def test_tex_and_bbl_extractor_consistency(self):
        """Test that both extractors handle similar inputs consistently."""
        # Create both TEX and BBL files with same citation keys
        tex_content = """\\cite{common_key1, common_key2}"""
        bbl_content = """
        \\bibitem{common_key1}
        First reference.
        \\bibitem{common_key2}
        Second reference.
        """

        tex_extractor = TexCitationExtractor()
        bbl_extractor = BBLCitationExtractor()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tex", delete=False
        ) as tex_f:
            tex_f.write(tex_content)
            tex_f.flush()
            tex_citations = tex_extractor.extract_citations(tex_f.name)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".bbl", delete=False
        ) as bbl_f:
            bbl_f.write(bbl_content)
            bbl_f.flush()
            bbl_citations = bbl_extractor.extract_citations(bbl_f.name)

        tex_keys = {c.key for c in tex_citations}
        bbl_keys = {c.key for c in bbl_citations}

        assert tex_keys == bbl_keys == {"common_key1", "common_key2"}

    def test_citation_info_sources_accumulation(self):
        """Test that multiple sources for same key are properly accumulated."""
        tex_content = """
        \\cite{shared_key}
        \\citep{shared_key}
        \\textcite{shared_key}
        """

        extractor = TexCitationExtractor()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as f:
            f.write(tex_content)
            f.flush()
            citations = extractor.extract_citations(f.name)

        assert len(citations) == 1
        assert citations[0].key == "shared_key"
        assert len(citations[0].sources) == 3

        # Verify all sources have correct line numbers
        line_numbers = [source.line_number for source in citations[0].sources]
        assert sorted(line_numbers) == [2, 3, 4]
