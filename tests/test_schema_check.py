"""Light JSON Schema subset. No model."""

from __future__ import annotations

import pytest

from prompt_eval.schema_check import assert_schema_shape, explain_mismatch


def test_bool_is_not_an_integer_or_number() -> None:
    integer_schema = {"type": "integer"}
    number_schema = {"type": "number"}
    assert explain_mismatch(1, integer_schema) is None
    assert explain_mismatch(True, integer_schema) is not None
    assert explain_mismatch(False, number_schema) is not None
    assert explain_mismatch(1.5, number_schema) is None
    assert explain_mismatch(1.5, integer_schema) is not None


def test_array_items_and_required_properties() -> None:
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}},
        },
    }
    assert explain_mismatch([{"name": "ada"}], schema) is None
    reason = explain_mismatch([{"name": 1}], schema)
    assert reason is not None
    assert "name" in reason
    missing = explain_mismatch([{}], schema)
    assert missing is not None
    assert "missing required" in missing


def test_unsupported_keywords_fail_at_shape_check() -> None:
    with pytest.raises(ValueError, match="unsupported keyword"):
        assert_schema_shape({"type": "string", "minLength": 1})
    with pytest.raises(ValueError, match="type is required"):
        assert_schema_shape({"enum": ["a"]})
