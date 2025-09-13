import logging
from typing import List, Dict
from .data_type import BibTypeRule
from .basic import DefaultRules
from .acm import ACMBibStyle

logger = logging.getLogger(__name__)


ExtraRuleSet = {
    "acm": ACMBibStyle,
}


class BibRuleRegister:
    def __init__(self, style: str = "default"):
        self._rules: set[BibTypeRule] = set(DefaultRules)
        if style != "default" and ExtraRuleSet.get(style, None) is not None:
            logger.warning(f"Loading extra rules for style: {style}")
            for rule in ExtraRuleSet[style]:
                self.register_rule(rule, force=True)

    def get_rule(self, bib_type: str) -> BibTypeRule:
        for rule in self._rules:
            if rule.entry_type == bib_type:
                return rule
        return self.get_rule("misc")

    def get_rules(self) -> List[BibTypeRule]:
        return list(self._rules)

    def get_named_rules(self) -> Dict[str, BibTypeRule]:
        rule_dict = {}
        for rule in self._rules:
            rule_dict[rule.entry_type] = rule
        return rule_dict

    def register_rule(self, rule: BibTypeRule, force: bool = False):
        if rule in self._rules:
            if force:
                self._rules.remove(rule)
            else:
                raise ValueError(
                    f"Rule for {rule.entry_type} ({rule.standard_name}) already registered"
                )
        self._rules.add(rule)

    def unregister_rule(self, rule: BibTypeRule):
        if rule in self._rules:
            self._rules.remove(rule)
        else:
            raise ValueError(
                f"Rule for {rule.entry_type} ({rule.standard_name}) not found"
            )

    def remove_rule(self, bib_type: str, standard_name: str = "default"):
        rule_to_remove = None
        for rule in self._rules:
            if rule.entry_type == bib_type and rule.standard_name == standard_name:
                rule_to_remove = rule
                break
        if rule_to_remove:
            self._rules.remove(rule_to_remove)
        else:
            raise ValueError(f"Rule for {bib_type} ({standard_name}) not found")


__all__ = ["BibRuleRegister", "BibTypeRule"]
