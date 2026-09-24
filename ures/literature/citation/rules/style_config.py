import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .data_type import BibTypeRule, FormattingRules, OutputRules

STYLES_DIR = Path(__file__).resolve().parent.parent / "styles"
PACKAGED_STYLES = ("acm", "ieee")
LOCAL_STYLE_FILENAME = "bibstyle.json"
PROTECTED_FIELDS = ("author", "title", "year")
PROFILES = ("library", "submission", "camera-ready")
OVERLAY_TYPE_KEYS = (
    "required_add",
    "required_remove",
    "suggested_add",
    "suggested_remove",
)


class StyleConfigError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def packaged_style_path(name: str) -> Path:
    if name not in PACKAGED_STYLES:
        raise StyleConfigError(f"unknown style '{name}' (use acm or ieee)")
    path = STYLES_DIR / f"{name}.json"
    if not path.is_file():
        raise StyleConfigError(f"packaged style missing: {path}")
    return path


def load_json_file(path: Path) -> Dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise StyleConfigError(f"failed to parse {path.name}: {exc.msg}") from exc
    except OSError as exc:
        raise StyleConfigError(f"failed to read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise StyleConfigError(f"{path.name} must be a JSON object")
    return data


def load_packaged_style(name: str) -> Dict[str, Any]:
    return load_json_file(packaged_style_path(name))


def _entry_types(data: Dict[str, Any]) -> Dict[str, Any]:
    types = data.get("entry_types", {})
    if not isinstance(types, dict):
        raise StyleConfigError("entry_types must be an object")
    return types


def _is_full_style(data: Dict[str, Any]) -> bool:
    if "extends" in data:
        return False
    types = _entry_types(data)
    if not types:
        return False
    return any(
        isinstance(spec, dict) and ("required" in spec or "suggested" in spec)
        for spec in types.values()
    )


def _is_overlay(data: Dict[str, Any]) -> bool:
    if "extends" in data:
        return True
    types = _entry_types(data)
    return any(
        isinstance(spec, dict) and any(key in spec for key in OVERLAY_TYPE_KEYS)
        for spec in types.values()
    )


def _dedupe(values: List[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def expand_field_spec(spec: str) -> List[str]:
    """Split a required/suggested spec into concrete BibTeX field names."""
    keys: List[str] = []
    for alternative in str(spec).split("|"):
        for part in alternative.split("+"):
            name = part.strip()
            if name and name not in keys:
                keys.append(name)
    return keys


def _apply_overlay(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    if "max_authors" in overlay:
        merged["max_authors"] = overlay["max_authors"]
    if "proceedings_style" in overlay:
        merged["proceedings_style"] = overlay["proceedings_style"]

    overlay_types = _entry_types(overlay)
    merged_types = _entry_types(merged)
    for entry_type, spec in overlay_types.items():
        if not isinstance(spec, dict):
            raise StyleConfigError(f"entry_types.{entry_type} must be an object")
        if any(key in spec for key in ("required", "suggested")):
            raise StyleConfigError(
                "bibstyle.json: cannot mix extends with complete required/suggested lists"
            )
        current = merged_types.get(
            entry_type,
            {
                "required": ["author", "title", "year"],
                "suggested": [],
            },
        )
        required = list(current.get("required", ["author", "title", "year"]))
        suggested = list(current.get("suggested", []))
        for field_name in spec.get("required_remove", []):
            if field_name in PROTECTED_FIELDS:
                raise StyleConfigError(
                    f"bibstyle.json: cannot remove required field {field_name}"
                )
            required = [item for item in required if item != field_name]
        for field_name in spec.get("suggested_remove", []):
            suggested = [item for item in suggested if item != field_name]
        required.extend(spec.get("required_add", []))
        suggested.extend(spec.get("suggested_add", []))
        required = _dedupe(required)
        suggested = _dedupe(suggested)
        removed_specs = list(spec.get("required_remove", [])) + list(
            spec.get("suggested_remove", [])
        )
        dropped: List[str] = []
        for field_spec in removed_specs:
            for key in expand_field_spec(field_spec):
                if key in PROTECTED_FIELDS:
                    raise StyleConfigError(
                        f"bibstyle.json: cannot remove required field {key}"
                    )
                dropped.append(key)
        kept = []
        for item in required + suggested:
            kept.extend(expand_field_spec(item))
        kept_names = set(kept)
        merged_types[entry_type] = {
            "required": required,
            "suggested": suggested,
            "dropped": [key for key in _dedupe(dropped) if key not in kept_names],
        }
    merged["entry_types"] = merged_types
    return merged


def resolve_style_document(
    local_path: Optional[Path] = None, style_name: Optional[str] = None
) -> Dict[str, Any]:
    if style_name is not None and local_path is None:
        data = load_packaged_style(style_name)
        data["name"] = style_name
        return data

    path = local_path or Path.cwd() / LOCAL_STYLE_FILENAME
    if not path.is_file():
        data = load_packaged_style(style_name or "acm")
        data["name"] = style_name or "acm"
        return data

    data = load_json_file(path)
    full = _is_full_style(data)
    overlay = _is_overlay(data)
    if full and overlay:
        raise StyleConfigError(
            "bibstyle.json: cannot mix extends with complete required/suggested lists"
        )
    if overlay:
        extends = data.get("extends", "acm")
        if extends not in PACKAGED_STYLES:
            raise StyleConfigError(f"unknown style '{extends}' (use acm or ieee)")
        merged = _apply_overlay(load_packaged_style(extends), data)
        merged["name"] = extends
        return merged
    if full:
        name = data.get("name", "acm")
        if name not in PACKAGED_STYLES:
            raise StyleConfigError(f"unknown style '{name}' (use acm or ieee)")
        data["name"] = name
        return data
    raise StyleConfigError(
        "bibstyle.json must be a full style (from init) or an extends overlay"
    )


def apply_profile(style: Dict[str, Any], profile: str) -> Dict[str, Any]:
    if profile not in PROFILES:
        raise StyleConfigError(
            f"unknown profile '{profile}' (use library, submission, or camera-ready)"
        )
    styled = copy.deepcopy(style)
    style_name = styled.get("name", "acm")
    types = _entry_types(styled)
    for spec in types.values():
        required = list(spec.get("required", []))
        suggested = list(spec.get("suggested", []))
        if profile == "submission":
            ident = "url" if style_name == "ieee" else "doi|url"
            if ident not in required:
                required.append(ident)
            suggested = [item for item in suggested if item != ident]
        elif profile == "camera-ready":
            for item in suggested:
                if item not in required:
                    required.append(item)
            suggested = []
        spec["required"] = required
        spec["suggested"] = suggested
    return styled


def style_to_rules(style: Dict[str, Any]) -> Tuple[List[BibTypeRule], int, str, str]:
    name = style.get("name", "acm")
    max_authors = int(style.get("max_authors", 0) or 0)
    proceedings_style = str(style.get("proceedings_style", "proceedings"))
    rules: List[BibTypeRule] = []
    for entry_type, spec in _entry_types(style).items():
        rules.append(
            BibTypeRule(
                entry_type=entry_type,
                standard_name=name,
                required_fields=list(spec.get("required", [])),
                suggested_fields=list(spec.get("suggested", [])),
                optional_fields=[],
                forbidden_fields=list(spec.get("dropped", [])),
                formatting=FormattingRules(proceedings_style=proceedings_style),
                output=OutputRules(max_authors=max_authors),
            )
        )
    return rules, max_authors, name, proceedings_style
