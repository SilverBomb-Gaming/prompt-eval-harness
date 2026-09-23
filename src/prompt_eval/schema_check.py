"""A small JSON Schema subset used by the json_schema check.

Supported keywords: type, required, properties, items, enum.
Supported types: object, array, string, number, integer, boolean, null.
"""

from __future__ import annotations

import json
import re

_ALLOWED_KEYS = {"type", "required", "properties", "items", "enum"}
_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", flags=re.DOTALL)


def assert_schema_shape(schema: object, path: str = "json_schema") -> None:
    """Reject keywords this harness does not implement. Called while loading a suite."""
    if not isinstance(schema, dict):
        raise ValueError(f"{path} must be a mapping")
    unknown = sorted(set(schema) - _ALLOWED_KEYS)
    if unknown:
        raise ValueError(f"{path}: unsupported keyword(s): {', '.join(unknown)}")
    if "type" not in schema:
        raise ValueError(f"{path}: type is required")
    expected = schema["type"]
    if not isinstance(expected, str) or expected not in _TYPES:
        raise ValueError(f"{path}: unsupported type {expected!r}")
    if "enum" in schema:
        enum_values = schema["enum"]
        if not isinstance(enum_values, list) or not enum_values:
            raise ValueError(f"{path}: enum must be a non-empty list")
    if "required" in schema:
        if expected != "object":
            raise ValueError(f"{path}: required is only valid for type object")
        required = schema["required"]
        if not isinstance(required, list) or not all(isinstance(item, str) and item for item in required):
            raise ValueError(f"{path}: required must be a list of non-empty strings")
    if "properties" in schema:
        if expected != "object":
            raise ValueError(f"{path}: properties is only valid for type object")
        properties = schema["properties"]
        if not isinstance(properties, dict):
            raise ValueError(f"{path}: properties must be a mapping")
        for key, subschema in properties.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{path}: property names must be non-empty strings")
            assert_schema_shape(subschema, f"{path}.{key}")
    if "items" in schema:
        if expected != "array":
            raise ValueError(f"{path}: items is only valid for type array")
        assert_schema_shape(schema["items"], f"{path}.items")


def explain_mismatch(instance: object, schema: dict[str, object], path: str = "$") -> str | None:
    """Return None when instance matches schema, otherwise a short reason."""
    expected = schema["type"]
    if not isinstance(expected, str) or not _type_ok(instance, expected):
        return f"{path}: expected {expected}"
    if "enum" in schema:
        enum_values = schema["enum"]
        if isinstance(enum_values, list) and instance not in enum_values:
            return f"{path}: expected one of {enum_values!r}"
    if expected == "object" and isinstance(instance, dict):
        required = schema.get("required", [])
        if isinstance(required, list):
            for key in required:
                if key not in instance:
                    return f"{path}: missing required property {key!r}"
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for key, subschema in properties.items():
                if key in instance and isinstance(subschema, dict):
                    reason = explain_mismatch(instance[key], subschema, f"{path}.{key}")
                    if reason:
                        return reason
    if expected == "array" and isinstance(instance, list) and isinstance(schema.get("items"), dict):
        items = schema["items"]
        assert isinstance(items, dict)
        for index, item in enumerate(instance):
            reason = explain_mismatch(item, items, f"{path}[{index}]")
            if reason:
                return reason
    return None


def parse_json_document(text: str) -> object:
    """Parse a model reply that is JSON, a fenced block, or a single {...} span.

    Raises ValueError when none of those parses.
    """
    stripped = text.strip()
    candidates: list[str] = []
    if stripped:
        candidates.append(stripped)
    fence = _FENCE.search(stripped)
    if fence:
        candidates.append(fence.group(1).strip())
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        candidates.append(stripped[start : end + 1])
    start_list = stripped.find("[")
    end_list = stripped.rfind("]")
    if start_list != -1 and end_list > start_list:
        candidates.append(stripped[start_list : end_list + 1])
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("response is not JSON")


def _type_ok(instance: object, expected: str) -> bool:
    if expected == "object":
        return isinstance(instance, dict)
    if expected == "array":
        return isinstance(instance, list)
    if expected == "string":
        return isinstance(instance, str)
    if expected == "boolean":
        return isinstance(instance, bool)
    if expected == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if expected == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if expected == "null":
        return instance is None
    return False
