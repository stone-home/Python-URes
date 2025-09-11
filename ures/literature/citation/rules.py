from abc import ABC, abstractmethod


class Rule(ABC):
    """
    Abstract base class for a citation style rule.
    Each rule must implement a check method.
    """

    @abstractmethod
    def get_required_fields(self, entry_type: str) -> list:
        # 获取特定文献类型所需的所有字段
        pass

    @abstractmethod
    def check(self, key: str, entry_content: str) -> list:
        # 检查文献条目是否符合规则，返回一个包含错误的列表
        pass


# --- Concrete Rule Implementations ---
class ACM_Rule(Rule):
    """
    Implements validation rules for ACM citation style.
    """

    def __init__(self):
        self.rules = {
            "article": {"required": ["author", "title", "journal", "year"]},
            "inproceedings": {"required": ["author", "title", "booktitle", "year"]},
            "book": {"required": ["author", "title", "publisher", "year"]},
            "phdthesis": {"required": ["author", "title", "school", "year"]},
            "mastersthesis": {"required": ["author", "title", "school", "year"]},
            "techreport": {"required": ["author", "title", "institution", "year"]},
            "misc": {"required": ["author", "title", "year"]},
            "webpage": {"required": ["author", "title", "url", "year"]},
        }

    def get_required_fields(self, entry_type: str) -> list:
        return self.rules.get(entry_type, {}).get("required", [])

    def check(self, key: str, entry_content: str) -> list:
        errors = []
        entry_type = self._infer_entry_type(entry_content)
        required_fields = self.get_required_fields(entry_type)

        for field in required_fields:
            if not self._has_field(field, entry_content):
                errors.append(f"缺少 '{field}' 字段")

        return errors

    def _infer_entry_type(self, entry_content: str) -> str:
        # 根据内容推断文献类型
        if (
            "inproceedings" in entry_content.lower()
            or "conference" in entry_content.lower()
        ):
            return "inproceedings"
        elif "book" in entry_content.lower():
            return "book"
        elif "arxiv" in entry_content.lower():
            return "misc"
        elif "webpage" in entry_content.lower():
            return "webpage"
        else:
            return "article"

    def _has_field(self, field: str, entry_content: str) -> bool:
        # 检查条目内容中是否包含指定字段
        if field == "author":
            return "bibfield{author}" in entry_content
        elif field == "title":
            return (
                "\\showarticletitle" in entry_content
                or "\\bibinfo{title}" in entry_content
            )
        elif field == "journal":
            return "\\bibinfo{journal}" in entry_content
        elif field == "booktitle":
            return "\\bibinfo{booktitle}" in entry_content
        elif field == "url":
            return "\\showURL" in entry_content
        else:
            return field in entry_content.lower()


class IEEE_Rule(Rule):
    """
    Implements validation rules for IEEE citation style.
    """

    def __init__(self):
        self.rules = {
            "article": {
                "required": [
                    "author",
                    "title",
                    "journal",
                    "volume",
                    "number",
                    "pages",
                    "month",
                    "year",
                ]
            },
            "inproceedings": {
                "required": ["author", "title", "booktitle", "year", "month", "pages"]
            },
            "book": {"required": ["author", "title", "publisher", "year"]},
            "phdthesis": {"required": ["author", "title", "school", "year"]},
            "mastersthesis": {"required": ["author", "title", "school", "year"]},
            "techreport": {"required": ["author", "title", "institution", "year"]},
            "misc": {"required": ["author", "title", "year"]},
            "webpage": {"required": ["author", "title", "url", "year"]},
        }

    def get_required_fields(self, entry_type: str) -> list:
        return self.rules.get(entry_type, {}).get("required", [])

    def check(self, key: str, entry_content: str) -> list:
        errors = []
        entry_type = self._infer_entry_type(entry_content)
        required_fields = self.get_required_fields(entry_type)

        for field in required_fields:
            if not self._has_field(field, entry_content):
                errors.append(f"缺少 '{field}' 字段")

        return errors

    def _infer_entry_type(self, entry_content: str) -> str:
        # 根据内容推断文献类型
        if (
            "inproceedings" in entry_content.lower()
            or "conference" in entry_content.lower()
        ):
            return "inproceedings"
        elif "book" in entry_content.lower():
            return "book"
        elif "arxiv" in entry_content.lower():
            return "misc"
        elif "webpage" in entry_content.lower():
            return "webpage"
        else:
            return "article"

    def _has_field(self, field: str, entry_content: str) -> bool:
        # 检查条目内容中是否包含指定字段
        if field == "author":
            return "bibfield{author}" in entry_content
        elif field == "title":
            return (
                "\\showarticletitle" in entry_content
                or "\\bibinfo{title}" in entry_content
            )
        elif field == "journal":
            return "\\bibinfo{journal}" in entry_content
        elif field == "booktitle":
            return "\\bibinfo{booktitle}" in entry_content
        elif field == "pages":
            return "\\bibinfo{pages}" in entry_content
        elif field == "volume":
            return "\\bibinfo{volume}" in entry_content
        elif field == "number":
            return "\\bibinfo{number}" in entry_content
        elif field == "month":
            return "\\bibinfo{month}" in entry_content
        elif field == "url":
            return "\\showURL" in entry_content
        else:
            return field in entry_content.lower()
