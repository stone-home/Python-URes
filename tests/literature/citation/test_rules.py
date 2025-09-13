import pytest
from unittest.mock import patch
from typing import List

from ures.literature.citation.rules.data_type import (
    BibTypeRule,
    FormattingRules,
    OutputRules,
)
from ures.literature.citation.rules.basic import DefaultRules, BasicRequiredFields
from ures.literature.citation.rules.acm import ACMBibStyle
from ures.literature.citation.rules import BibRuleRegister, ExtraRuleSet


class TestBasicConfiguration:
    """Test cases for basic configuration validation"""

    def test_basic_required_fields_type(self):
        """Test that BasicRequiredFields is a list of strings"""
        assert isinstance(BasicRequiredFields, list)
        assert all(isinstance(field, str) for field in BasicRequiredFields)
        assert len(BasicRequiredFields) > 0

    def test_basic_required_fields_content(self):
        """Test that BasicRequiredFields contains expected fields"""
        expected_fields = {"title", "year", "month", "author"}
        assert set(BasicRequiredFields) == expected_fields

    def test_default_rules_type(self):
        """Test that DefaultRules is a list of BibTypeRule instances"""
        assert isinstance(DefaultRules, list)
        assert all(isinstance(rule, BibTypeRule) for rule in DefaultRules)
        assert len(DefaultRules) > 0

    def test_default_rules_attributes(self):
        """Test that all DefaultRules have proper BibTypeRule attributes"""
        for rule in DefaultRules:
            # Test required attributes exist
            assert hasattr(rule, "entry_type")
            assert hasattr(rule, "standard_name")
            assert hasattr(rule, "required_fields")
            assert hasattr(rule, "optional_fields")
            assert hasattr(rule, "forbidden_fields")
            assert hasattr(rule, "field_mappings")
            assert hasattr(rule, "formatting")
            assert hasattr(rule, "output")

            # Test attribute types
            assert isinstance(rule.entry_type, str)
            assert isinstance(rule.standard_name, str)
            assert isinstance(rule.required_fields, list)
            assert isinstance(rule.optional_fields, list)
            assert isinstance(rule.forbidden_fields, list)
            assert isinstance(rule.field_mappings, dict)
            assert isinstance(rule.formatting, FormattingRules)
            assert isinstance(rule.output, OutputRules)

            # Test that entry_type is not empty
            assert rule.entry_type.strip() != ""

    def test_default_rules_standard_name(self):
        """Test that all DefaultRules have 'default' as standard_name"""
        for rule in DefaultRules:
            assert rule.standard_name == "default"

    def test_default_rules_unique_entry_types(self):
        """Test that all entry types in DefaultRules are unique"""
        entry_types = [rule.entry_type for rule in DefaultRules]
        assert len(entry_types) == len(set(entry_types))


class TestACMConfiguration:
    """Test cases for ACM configuration validation"""

    def test_acm_bib_style_type(self):
        """Test that ACMBibStyle is a list of BibTypeRule instances"""
        assert isinstance(ACMBibStyle, list)
        assert all(isinstance(rule, BibTypeRule) for rule in ACMBibStyle)
        assert len(ACMBibStyle) > 0

    def test_acm_bib_style_attributes(self):
        """Test that all ACMBibStyle rules have proper BibTypeRule attributes"""
        for rule in ACMBibStyle:
            # Test required attributes exist
            assert hasattr(rule, "entry_type")
            assert hasattr(rule, "standard_name")
            assert hasattr(rule, "required_fields")
            assert hasattr(rule, "optional_fields")
            assert hasattr(rule, "forbidden_fields")
            assert hasattr(rule, "field_mappings")
            assert hasattr(rule, "formatting")
            assert hasattr(rule, "output")

            # Test attribute types
            assert isinstance(rule.entry_type, str)
            assert isinstance(rule.standard_name, str)
            assert isinstance(rule.required_fields, list)
            assert isinstance(rule.optional_fields, list)
            assert isinstance(rule.forbidden_fields, list)
            assert isinstance(rule.field_mappings, dict)
            assert isinstance(rule.formatting, FormattingRules)
            assert isinstance(rule.output, OutputRules)

            # Test that entry_type is not empty
            assert rule.entry_type.strip() != ""

    def test_acm_bib_style_standard_name(self):
        """Test that all ACMBibStyle rules have 'acm' as standard_name"""
        for rule in ACMBibStyle:
            assert rule.standard_name == "acm"

    def test_acm_bib_style_unique_entry_types(self):
        """Test that all entry types in ACMBibStyle are unique"""
        entry_types = [rule.entry_type for rule in ACMBibStyle]
        assert len(entry_types) == len(set(entry_types))

    def test_acm_bib_style_required_fields_include_basic(self):
        """Test that ACM rules include BasicRequiredFields in their required fields"""
        for rule in ACMBibStyle:
            basic_fields_set = set(BasicRequiredFields)
            rule_fields_set = set(rule.required_fields)
            assert basic_fields_set.issubset(rule_fields_set), (
                f"Rule {rule.entry_type} missing basic required fields: "
                f"{basic_fields_set - rule_fields_set}"
            )


class TestBibTypeRuleValidation:
    """Test cases for BibTypeRule class validation"""

    def test_bib_type_rule_creation(self):
        """Test creating a valid BibTypeRule instance"""
        rule = BibTypeRule(
            entry_type="test",
            standard_name="test_standard",
            required_fields=["title", "author"],
            optional_fields=["year"],
            forbidden_fields=["note"],
            field_mappings={"old_field": "new_field"},
        )

        assert rule.entry_type == "test"
        assert rule.standard_name == "test_standard"
        assert rule.required_fields == ["title", "author"]
        assert rule.optional_fields == ["year"]
        assert rule.forbidden_fields == ["note"]
        assert rule.field_mappings == {"old_field": "new_field"}

    def test_bib_type_rule_empty_entry_type_raises_error(self):
        """Test that empty entry_type raises ValueError"""
        with pytest.raises(ValueError, match="Entry type cannot be empty"):
            BibTypeRule(entry_type="")

    def test_bib_type_rule_default_values(self):
        """Test BibTypeRule default values"""
        rule = BibTypeRule(entry_type="test")

        assert rule.entry_type == "test"
        assert rule.standard_name == "default"
        assert rule.required_fields == []
        assert rule.optional_fields == []
        assert rule.forbidden_fields == []
        assert rule.field_mappings == {}
        assert isinstance(rule.formatting, FormattingRules)
        assert isinstance(rule.output, OutputRules)
        assert rule.version == "1.0"

    def test_bib_type_rule_equality(self):
        """Test BibTypeRule equality comparison"""
        rule1 = BibTypeRule(entry_type="test", version="1.0")
        rule2 = BibTypeRule(entry_type="test", version="1.0")
        rule3 = BibTypeRule(entry_type="different", version="1.0")
        rule4 = BibTypeRule(entry_type="test", version="2.0")

        assert rule1 == rule2
        assert rule1 != rule3
        assert rule1 != rule4

    def test_bib_type_rule_hash(self):
        """Test BibTypeRule hashing"""
        rule1 = BibTypeRule(entry_type="test")
        rule2 = BibTypeRule(entry_type="test")
        rule3 = BibTypeRule(entry_type="different")

        assert hash(rule1) == hash(rule2)
        assert hash(rule1) != hash(rule3)

    def test_bib_type_rule_to_dict(self):
        """Test BibTypeRule to_dict method"""
        rule = BibTypeRule(
            entry_type="test", required_fields=["title"], optional_fields=["year"]
        )

        rule_dict = rule.to_dict()
        assert isinstance(rule_dict, dict)
        assert rule_dict["entry_type"] == "test"
        assert rule_dict["required_fields"] == ["title"]
        assert rule_dict["optional_fields"] == ["year"]

    def test_bib_type_rule_repr(self):
        """Test BibTypeRule string representation"""
        rule = BibTypeRule(entry_type="test", standard_name="custom", version="2.0")
        repr_str = repr(rule)

        assert "BibTypeRule(test[custom], version=2.0)" == repr_str


class TestBibRuleRegister:
    """Test cases for BibRuleRegister functionality"""

    def test_register_initialization_default(self):
        """Test BibRuleRegister initialization with default style"""
        register = BibRuleRegister()

        rules = register.get_rules()
        assert len(rules) == len(DefaultRules)
        assert all(rule.standard_name == "default" for rule in rules)

    def test_register_initialization_acm_style(self):
        """Test BibRuleRegister initialization with ACM style"""
        register = BibRuleRegister(style="acm")

        rules = register.get_rules()
        for rule in ACMBibStyle:
            assert rule in rules

    def test_get_rule_existing(self):
        """Test getting an existing rule"""
        register = BibRuleRegister()

        article_rule = register.get_rule("article")
        assert article_rule.entry_type == "article"
        assert article_rule.standard_name == "default"

    def test_get_rule_nonexistent_returns_misc(self):
        """Test getting a non-existent rule returns misc rule"""
        register = BibRuleRegister()

        nonexistent_rule = register.get_rule("nonexistent_type")
        assert nonexistent_rule.entry_type == "misc"

    def test_get_named_rules(self):
        """Test getting rules as a dictionary"""
        register = BibRuleRegister()

        named_rules = register.get_named_rules()
        assert isinstance(named_rules, dict)

        # Check that all default entry types are present
        default_entry_types = {rule.entry_type for rule in DefaultRules}
        named_rule_types = set(named_rules.keys())
        assert default_entry_types == named_rule_types

    def test_register_new_rule(self):
        """Test registering a new rule"""
        register = BibRuleRegister()
        initial_count = len(register.get_rules())

        new_rule = BibTypeRule(entry_type="custom", standard_name="test")
        register.register_rule(new_rule)

        assert len(register.get_rules()) == initial_count + 1
        assert register.get_rule("custom") == new_rule

    def test_register_duplicate_rule_without_force_raises_error(self):
        """Test registering duplicate rule without force raises ValueError"""
        register = BibRuleRegister()

        duplicate_rule = BibTypeRule(entry_type="article", standard_name="default")

        with pytest.raises(
            ValueError, match="Rule for article \\(default\\) already registered"
        ):
            register.register_rule(duplicate_rule)

    def test_register_duplicate_rule_with_force_replaces(self):
        """Test registering duplicate rule with force replaces existing rule"""
        register = BibRuleRegister()
        initial_count = len(register.get_rules())

        new_rule = BibTypeRule(
            entry_type="article",
            standard_name="default",
            required_fields=["custom_field"],
        )
        register.register_rule(new_rule, force=True)

        assert len(register.get_rules()) == initial_count  # Same count
        retrieved_rule = register.get_rule("article")
        assert "custom_field" in retrieved_rule.required_fields

    def test_unregister_existing_rule(self):
        """Test unregistering an existing rule"""
        register = BibRuleRegister()
        initial_count = len(register.get_rules())

        article_rule = register.get_rule("article")
        register.unregister_rule(article_rule)

        assert len(register.get_rules()) == initial_count - 1
        # Should now return misc rule when asking for article
        assert register.get_rule("article").entry_type == "misc"

    def test_unregister_nonexistent_rule_raises_error(self):
        """Test unregistering non-existent rule raises ValueError"""
        register = BibRuleRegister()

        nonexistent_rule = BibTypeRule(entry_type="nonexistent", standard_name="test")

        with pytest.raises(
            ValueError, match="Rule for nonexistent \\(test\\) not found"
        ):
            register.unregister_rule(nonexistent_rule)

    def test_remove_rule_by_type_and_name(self):
        """Test removing rule by entry type and standard name"""
        register = BibRuleRegister()
        initial_count = len(register.get_rules())

        register.remove_rule("article", "default")

        assert len(register.get_rules()) == initial_count - 1
        # Should now return misc rule when asking for article
        assert register.get_rule("article").entry_type == "misc"

    def test_remove_nonexistent_rule_raises_error(self):
        """Test removing non-existent rule raises ValueError"""
        register = BibRuleRegister()

        with pytest.raises(
            ValueError, match="Rule for nonexistent \\(default\\) not found"
        ):
            register.remove_rule("nonexistent", "default")


class TestExtraRuleSet:
    """Test cases for ExtraRuleSet configuration"""

    def test_extra_rule_set_structure(self):
        """Test that ExtraRuleSet has proper structure"""
        assert isinstance(ExtraRuleSet, dict)
        assert "acm" in ExtraRuleSet
        assert isinstance(ExtraRuleSet["acm"], list)

    def test_extra_rule_set_acm_contains_bib_type_rules(self):
        """Test that ExtraRuleSet['acm'] contains BibTypeRule instances"""
        acm_rules = ExtraRuleSet["acm"]
        assert all(isinstance(rule, BibTypeRule) for rule in acm_rules)
        assert len(acm_rules) > 0
