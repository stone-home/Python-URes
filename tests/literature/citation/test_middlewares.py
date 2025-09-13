from unittest.mock import Mock, patch
from bibtexparser.model import Entry, Field
from bibtexparser.middlewares import NameParts

from ures.literature.citation.middlewares import (
    FieldNormalizationMiddleware,
    LanguageAsciiNormalizationMiddleware,
    DateSpiltToYearMonthDayMiddleware,
    PublisherNormalizationMiddleware,
    ProceedingsNormalizationMiddleware,
    TypeNormalizationMiddleware,
    RuleBasedValidationMiddleware,
    OutputCleanupNoneResultMiddleware,
    OutputOnlyDesiredFieldsMiddleware,
    OutputLimitMaxAuthors,
)
from ures.literature.citation.rules import BibRuleRegister
from ures.literature.citation.rules.data_type import (
    BibTypeRule,
    FormattingRules,
    OutputRules,
)


class TestFieldNormalizationMiddleware:
    """Test field normalization functionality."""

    def setup_method(self):
        self.rule_register = Mock()
        self.rule_register.get_rule.return_value = Mock(field_mappings={})
        self.rule_register.get_default_field_mapping = Mock(
            return_value={
                "rights": "copyright",
                "location": "address",
                "journaltitle": "journal",
                "titleaddon": "journal",
                "venue": "booktitle",  # Sometimes used for conference venue
                "langid": "language",
            }
        )
        self.middleware = FieldNormalizationMiddleware(self.rule_register)

    def test_field_mapping_basic(self):
        """Test basic field mapping transformations."""
        entry = Entry(
            key="test_entry",
            entry_type="article",
            fields=[
                Field(key="journaltitle", value="Nature"),
                Field(key="location", value="New York"),
                Field(key="langid", value="en"),
            ],
        )

        result = self.middleware.transform_entry(entry)
        print(result)

        field_keys = [field.key for field in result.fields]
        print(field_keys)
        assert "journal" in field_keys
        assert "address" in field_keys
        assert "language" in field_keys
        assert "journaltitle" not in field_keys
        assert "location" not in field_keys
        assert "langid" not in field_keys

    def test_pages_normalization_single_dash(self):
        """Test page range normalization with single dash."""
        entry = Entry(
            key="test_entry",
            entry_type="article",
            fields=[Field(key="pages", value="123-456")],
        )

        result = self.middleware.transform_entry(entry)
        pages_field = next(field for field in result.fields if field.key == "pages")
        assert pages_field.value == "123--456"

    def test_pages_normalization_unicode_dashes(self):
        """Test page range normalization with unicode dashes."""
        entry = Entry(
            key="test_entry",
            entry_type="article",
            fields=[Field(key="pages", value="123--456")],
        )

        result = self.middleware.transform_entry(entry)
        pages_field = next(field for field in result.fields if field.key == "pages")
        assert pages_field.value == "123--456"

    def test_pages_normalization_already_double_dash(self):
        """Test page range normalization when already using double dash."""
        entry = Entry(
            key="test_entry",
            entry_type="article",
            fields=[Field(key="pages", value="123--456")],
        )

        result = self.middleware.transform_entry(entry)
        pages_field = next(field for field in result.fields if field.key == "pages")
        assert pages_field.value == "123--456"

    def test_pages_normalization_non_string(self):
        """Test page range normalization with non-string input."""
        entry = Entry(
            key="test_entry",
            entry_type="article",
            fields=[Field(key="pages", value=123)],
        )

        result = self.middleware.transform_entry(entry)
        pages_field = next(field for field in result.fields if field.key == "pages")
        assert pages_field.value == "123"

    def test_custom_field_mappings_from_rules(self):
        """Test field mappings from rule register."""
        custom_rule = Mock()
        custom_rule.field_mappings = {"customfield": "standardfield"}
        self.rule_register.get_rule.return_value = custom_rule

        entry = Entry(
            key="test_entry",
            entry_type="article",
            fields=[Field(key="customfield", value="test_value")],
        )

        result = self.middleware.transform_entry(entry)
        field_keys = [field.key for field in result.fields]
        assert "standardfield" in field_keys
        assert "customfield" not in field_keys


class TestLanguageNormalizationMiddleware:
    """Test language normalization functionality."""

    def setup_method(self):
        self.middleware = LanguageAsciiNormalizationMiddleware()

    def test_special_cases_normalization(self):
        """Test special case language normalization."""
        entry = Entry(
            key="test_entry",
            entry_type="article",
            fields=[Field(key="langid", value="pinyin")],
        )

        result = self.middleware.transform_entry(entry)
        langid_field = next(field for field in result.fields if field.key == "langid")
        assert langid_field.value == "zh"

    def test_pycountry_language_lookup(self):
        """Test language lookup using pycountry."""
        entry = Entry(
            key="test_entry",
            entry_type="article",
            fields=[Field(key="langid", value="english")],
        )
        result = self.middleware.transform_entry(entry)
        assert result.get("langid").value == "en"

    def test_non_string_language_input(self):
        """Test non-string language input handling."""
        entry = Entry(
            key="test_entry",
            entry_type="article",
            fields=[Field(key="langid", value=123)],
        )

        result = self.middleware.transform_entry(entry)
        langid_field = next(field for field in result.fields if field.key == "langid")
        assert langid_field.value == 123

    def test_language_not_found(self):
        """Test language normalization when language not found."""
        entry = Entry(
            key="test_entry",
            entry_type="article",
            fields=[Field(key="langid", value="unknownlang")],
        )

        result = self.middleware.transform_entry(entry)
        langid_field = next(field for field in result.fields if field.key == "langid")
        assert langid_field.value == "unknownlang"


class TestDateProcessingMiddleware:
    """Test date processing functionality."""

    def setup_method(self):
        self.middleware = DateSpiltToYearMonthDayMiddleware()

    @patch("ures.literature.citation.middlewares.string2date")
    def test_date_split_to_components(self, mock_string2date):
        """Test splitting date field into year, month, day components."""
        mock_string2date.return_value = {
            "year": "2023",
            "month": "January",
            "day": "15",
        }

        entry = Entry(
            key="test_entry",
            entry_type="article",
            fields=[Field(key="date", value="2023-01-15")],
        )

        result = self.middleware.transform_entry(entry)

        field_dict = {field.key: field.value for field in result.fields}
        assert "date" not in field_dict
        assert field_dict["year"] == "2023"
        assert field_dict["month"] == "january"
        assert field_dict["day"] == "15"

    def test_no_date_field(self):
        """Test behavior when no date field is present."""
        entry = Entry(
            key="test_entry",
            entry_type="article",
            fields=[Field(key="title", value="Test Title")],
        )

        result = self.middleware.transform_entry(entry)
        field_keys = [field.key for field in result.fields]
        assert "year" not in field_keys
        assert "month" not in field_keys
        assert "day" not in field_keys


class TestPublisherNormalizationMiddleware:
    """Test publisher normalization functionality."""

    def setup_method(self):
        self.middleware = PublisherNormalizationMiddleware()

    def test_ieee_publisher_normalization_inproceedings(self):
        """Test IEEE publisher normalization for inproceedings."""
        entry = Entry(
            key="test_entry",
            entry_type="inproceedings",
            fields=[Field(key="publisher", value="ieee")],
        )

        result = self.middleware.transform_entry(entry)
        publisher_field = next(
            field for field in result.fields if field.key == "publisher"
        )
        assert publisher_field.value == "ieee Inc."

    def test_acm_publisher_normalization_article(self):
        """Test ACM publisher normalization for article."""
        entry = Entry(
            key="test_entry",
            entry_type="article",
            fields=[Field(key="publisher", value="{acm}")],
        )

        result = self.middleware.transform_entry(entry)
        publisher_field = next(
            field for field in result.fields if field.key == "publisher"
        )
        assert publisher_field.value == "{acm} Inc."

    def test_publisher_no_normalization_book(self):
        """Test publisher normalization is not applied to book entries."""
        entry = Entry(
            key="test_entry",
            entry_type="book",
            fields=[Field(key="publisher", value="ieee")],
        )

        result = self.middleware.transform_entry(entry)
        publisher_field = next(
            field for field in result.fields if field.key == "publisher"
        )
        assert publisher_field.value == "ieee"

    def test_publisher_no_normalization_other_publisher(self):
        """Test publisher normalization is not applied to other publishers."""
        entry = Entry(
            key="test_entry",
            entry_type="inproceedings",
            fields=[Field(key="publisher", value="Springer")],
        )

        result = self.middleware.transform_entry(entry)
        publisher_field = next(
            field for field in result.fields if field.key == "publisher"
        )
        assert publisher_field.value == "Springer"


class TestProceedingsNormalizationMiddleware:
    """Test proceedings normalization functionality."""

    def setup_method(self):
        self.rule_register = Mock()
        mock_rule = Mock()
        mock_rule.formatting.proceedings_style = "full"
        self.rule_register.get_rule.return_value = mock_rule
        self.middleware = ProceedingsNormalizationMiddleware(self.rule_register)

    def test_proceedings_prefix_removal(self):
        """Test removal of common proceedings prefixes."""
        test_cases = [
            (
                "In Proceedings of the International Conference",
                "International Conference",
            ),
            ("Proceedings of the Workshop", "Workshop"),
            ("In Proc. of the Symposium", "Symposium"),
            ("Proc. of International Meeting", "International Meeting"),
        ]

        for input_title, expected_base in test_cases:
            entry = Entry(
                key="test_entry",
                entry_type="inproceedings",
                fields=[Field(key="booktitle", value=input_title)],
            )

            result = self.middleware.transform_entry(entry)
            booktitle_field = next(
                field for field in result.fields if field.key == "booktitle"
            )
            assert booktitle_field.value == f"In Proceedings of the {expected_base}"

    def test_proceedings_style_proceedings(self):
        """Test proceedings style formatting."""
        mock_rule = Mock()
        mock_rule.formatting.proceedings_style = "proceedings"
        self.rule_register.get_rule.return_value = mock_rule

        entry = Entry(
            key="test_entry",
            entry_type="inproceedings",
            fields=[Field(key="booktitle", value="In Proceedings of the Conference")],
        )

        result = self.middleware.transform_entry(entry)
        booktitle_field = next(
            field for field in result.fields if field.key == "booktitle"
        )
        assert booktitle_field.value == "Proceedings of the Conference"

    def test_proceedings_style_minimal(self):
        """Test minimal proceedings style formatting."""
        mock_rule = Mock()
        mock_rule.formatting.proceedings_style = "minimal"
        self.rule_register.get_rule.return_value = mock_rule

        entry = Entry(
            key="test_entry",
            entry_type="inproceedings",
            fields=[Field(key="booktitle", value="Proceedings of the Workshop")],
        )

        result = self.middleware.transform_entry(entry)
        booktitle_field = next(
            field for field in result.fields if field.key == "booktitle"
        )
        assert booktitle_field.value == "In Workshop"

    def test_proceedings_style_remove(self):
        """Test remove proceedings style formatting."""
        mock_rule = Mock()
        mock_rule.formatting.proceedings_style = "remove"
        self.rule_register.get_rule.return_value = mock_rule

        entry = Entry(
            key="test_entry",
            entry_type="inproceedings",
            fields=[Field(key="booktitle", value="In Proceedings of the Summit")],
        )

        result = self.middleware.transform_entry(entry)
        booktitle_field = next(
            field for field in result.fields if field.key == "booktitle"
        )
        assert booktitle_field.value == "Summit"

    def test_non_inproceedings_entry_unchanged(self):
        """Test that non-inproceedings entries are not affected."""
        entry = Entry(
            key="test_entry",
            entry_type="article",
            fields=[Field(key="booktitle", value="In Proceedings of the Conference")],
        )

        result = self.middleware.transform_entry(entry)
        booktitle_field = next(
            field for field in result.fields if field.key == "booktitle"
        )
        assert booktitle_field.value == "In Proceedings of the Conference"


class TestTypeNormalizationMiddleware:
    """Test entry type normalization functionality."""

    def setup_method(self):
        self.middleware = TypeNormalizationMiddleware()

    def test_conference_to_inproceedings(self):
        """Test conference type normalization to inproceedings."""
        entry = Entry(
            key="test_entry",
            entry_type="conference",
            fields=[Field(key="title", value="Test Title")],
        )

        result = self.middleware.transform_entry(entry)
        assert result.entry_type == "inproceedings"

    def test_thesis_type_normalization(self):
        """Test thesis type normalization."""
        test_cases = [
            ("mastersthesis", "thesis"),
            ("phdthesis", "thesis"),
        ]

        for input_type, expected_type in test_cases:
            entry = Entry(
                key="test_entry",
                entry_type=input_type,
                fields=[Field(key="title", value="Test Title")],
            )

            result = self.middleware.transform_entry(entry)
            assert result.entry_type == expected_type

    def test_online_type_normalization(self):
        """Test online type normalization."""
        test_cases = [
            ("webpage", "online"),
            ("electronic", "online"),
        ]

        for input_type, expected_type in test_cases:
            entry = Entry(
                key="test_entry",
                entry_type=input_type,
                fields=[Field(key="title", value="Test Title")],
            )

            result = self.middleware.transform_entry(entry)
            assert result.entry_type == expected_type

    def test_unknown_type_unchanged(self):
        """Test that unknown types remain unchanged."""
        entry = Entry(
            key="test_entry",
            entry_type="unknown_type",
            fields=[Field(key="title", value="Test Title")],
        )

        result = self.middleware.transform_entry(entry)
        assert result.entry_type == "unknown_type"


class TestValidationMiddleware:
    """Test validation functionality."""

    def setup_method(self):
        self.rule_register = Mock()
        mock_rule = Mock()
        mock_rule.required_fields = ["title", "author", "year"]
        self.rule_register.get_rule.return_value = mock_rule
        self.middleware = RuleBasedValidationMiddleware(self.rule_register)

    def test_valid_entry(self):
        """Test validation of valid entry."""
        entry = Entry(
            key="test_entry",
            entry_type="article",
            fields=[
                Field(key="title", value="Test Title"),
                Field(key="author", value="Test Author"),
                Field(key="year", value="2023"),
            ],
        )

        result = self.middleware.transform_entry(entry)
        field_dict = {field.key: field.value for field in result.fields}
        assert field_dict["is_valid"] is True
        assert field_dict["missing_fields"] == []

    def test_invalid_entry_missing_fields(self):
        """Test validation of entry with missing required fields."""
        entry = Entry(
            key="test_entry",
            entry_type="article",
            fields=[
                Field(key="title", value="Test Title"),
                Field(key="year", value="2023"),
            ],
        )

        result = self.middleware.transform_entry(entry)
        field_dict = {field.key: field.value for field in result.fields}
        assert field_dict["is_valid"] is False
        assert "author" in field_dict["missing_fields"]

    def test_invalid_entry_empty_values(self):
        """Test validation of entry with empty field values."""
        entry = Entry(
            key="test_entry",
            entry_type="article",
            fields=[
                Field(key="title", value="Test Title"),
                Field(key="author", value=""),
                Field(key="year", value="2023"),
            ],
        )

        result = self.middleware.transform_entry(entry)
        field_dict = {field.key: field.value for field in result.fields}
        assert field_dict["is_valid"] is False
        assert "author" in field_dict["missing_fields"]


class TestOutputMiddlewares:
    """Test output processing middlewares."""

    def setup_method(self):
        self.rule_register = Mock()

    def test_cleanup_none_result_middleware(self):
        """Test cleanup of None and empty values."""
        middleware = OutputCleanupNoneResultMiddleware()
        entry = Entry(
            key="test_entry",
            entry_type="article",
            fields=[
                Field(key="title", value="Test Title"),
                Field(key="author", value=""),
                Field(key="year", value=None),
                Field(key="volume", value=[]),
                Field(key="journal", value="Test Journal"),
            ],
        )

        result = middleware.transform_entry(entry)
        field_keys = [field.key for field in result.fields]
        assert "title" in field_keys
        assert "journal" in field_keys
        assert "author" not in field_keys
        assert "year" not in field_keys
        assert "volume" not in field_keys

    def test_output_only_desired_fields_middleware(self):
        """Test filtering to only desired fields."""
        mock_rule = Mock()
        mock_rule.required_fields = ["title", "author"]
        mock_rule.optional_fields = ["year"]
        mock_rule.forbidden_fields = ["note"]
        mock_rule.output.required_middlewares = []
        self.rule_register.get_rule.return_value = mock_rule

        middleware = OutputOnlyDesiredFieldsMiddleware(self.rule_register)
        entry = Entry(
            key="test_entry",
            entry_type="article",
            fields=[
                Field(key="title", value="Test Title"),
                Field(key="author", value="Test Author"),
                Field(key="year", value="2023"),
                Field(key="note", value="Test Note"),
                Field(key="extra", value="Extra Field"),
                Field(key="is_valid", value=True),
            ],
        )

        result = middleware.transform_entry(entry)
        field_keys = [field.key for field in result.fields]
        assert "title" in field_keys
        assert "author" in field_keys
        assert "year" in field_keys
        assert "note" not in field_keys
        assert "extra" not in field_keys
        assert "is_valid" not in field_keys

    def test_output_limit_max_authors(self):
        """Test limiting maximum number of authors."""
        mock_rule = Mock()
        mock_rule.output.max_authors = 3
        self.rule_register.get_rule.return_value = mock_rule

        middleware = OutputLimitMaxAuthors(self.rule_register)

        # Create author list with NameParts
        authors = [
            NameParts(first=["John"], last=["Doe"]),
            NameParts(first=["Jane"], last=["Smith"]),
            NameParts(first=["Bob"], last=["Johnson"]),
            NameParts(first=["Alice"], last=["Brown"]),
            NameParts(first=["Charlie"], last=["Wilson"]),
        ]

        entry = Entry(
            key="test_entry",
            entry_type="article",
            fields=[
                Field(key="title", value="Test Title"),
                Field(key="author", value=authors),
            ],
        )

        result = middleware.transform_entry(entry)
        author_field = next(field for field in result.fields if field.key == "author")

        assert len(author_field.value) == 4  # 3 authors + "others"
        assert author_field.value[-1].first == ["others"]
        assert author_field.value[-1].last == []

    def test_output_limit_max_authors_under_limit(self):
        """Test max authors when under the limit."""
        mock_rule = Mock()
        mock_rule.output.max_authors = 5
        self.rule_register.get_rule.return_value = mock_rule

        middleware = OutputLimitMaxAuthors(self.rule_register)

        authors = [
            NameParts(first=["John"], last=["Doe"]),
            NameParts(first=["Jane"], last=["Smith"]),
        ]

        entry = Entry(
            key="test_entry",
            entry_type="article",
            fields=[
                Field(key="title", value="Test Title"),
                Field(key="author", value=authors),
            ],
        )

        result = middleware.transform_entry(entry)
        author_field = next(field for field in result.fields if field.key == "author")

        assert len(author_field.value) == 2
        assert all(author.first != ["others"] for author in author_field.value)


class TestMiddlewareIntegration:
    """Test middleware integration and edge cases."""

    def test_middleware_with_default_rule_register(self):
        """Test middleware creation with default rule register."""
        middleware = FieldNormalizationMiddleware()
        assert middleware.rule_register is not None

    def test_middleware_inheritance(self):
        """Test that all middlewares inherit from CitationMiddlewarePipeline."""
        from ures.literature.citation.middlewares import CitationMiddleware

        middlewares = [
            FieldNormalizationMiddleware(),
            LanguageAsciiNormalizationMiddleware(),
            DateSpiltToYearMonthDayMiddleware(),
            PublisherNormalizationMiddleware(),
            ProceedingsNormalizationMiddleware(),
            TypeNormalizationMiddleware(),
            RuleBasedValidationMiddleware(),
            OutputCleanupNoneResultMiddleware(),
            OutputOnlyDesiredFieldsMiddleware(),
            OutputLimitMaxAuthors(),
        ]

        for middleware in middlewares:
            assert isinstance(middleware, CitationMiddleware)

    def test_entry_deep_copy_preservation(self):
        """Test that entry modifications don't affect original entry."""
        middleware = TypeNormalizationMiddleware()
        original_entry = Entry(
            key="test_entry",
            entry_type="conference",
            fields=[Field(key="title", value="Test Title")],
        )

        original_type = original_entry.entry_type
        assert original_entry.entry_type == original_type

        result = middleware.transform_entry(original_entry)
        assert result.entry_type == "inproceedings"
