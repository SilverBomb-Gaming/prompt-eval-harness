"""Deterministic scorers. No model."""

from __future__ import annotations

from prompt_eval.models import CheckSpec
from prompt_eval.scorers import score_output


def test_contains_not_contains_and_regex_are_case_sensitive_except_pattern_flags() -> None:
    checks = CheckSpec.model_validate(
        {
            "contains": ["30 days"],
            "not_contains": ["lifetime warranty"],
            "regex": ["(?i)refunds are available"],
        }
    )
    passed = score_output("Refunds are available for 30 days.", 10, checks)
    assert [item.passed for item in passed] == [True, True, True]

    failed = score_output("refunds are available for thirty days.", 10, checks)
    assert [item.kind for item in failed] == ["contains", "not_contains", "regex"]
    assert failed[0].passed is False
    assert "missing substring" in (failed[0].detail or "")
    assert failed[1].passed is True
    assert failed[2].passed is True

    forbidden = score_output("Refunds are available for 30 days. lifetime warranty", 10, checks)
    assert forbidden[1].passed is False
    assert "forbidden" in (forbidden[1].detail or "")


def test_max_latency_uses_the_measured_value() -> None:
    checks = CheckSpec.model_validate({"max_latency_ms": 100})
    fast = score_output("ok", 100, checks)
    slow = score_output("ok", 100.1, checks)
    assert fast[0].passed is True
    assert slow[0].passed is False
    assert slow[0].detail is not None
    assert "100 ms" in slow[0].detail


def test_json_schema_parses_fences_and_fails_closed() -> None:
    checks = CheckSpec.model_validate(
        {
            "json_schema": {
                "type": "object",
                "required": ["priority", "summary"],
                "properties": {
                    "priority": {"type": "string", "enum": ["low"]},
                    "summary": {"type": "string"},
                },
            }
        }
    )
    raw = '```json\n{"priority": "low", "summary": "password reset"}\n```'
    assert score_output(raw, 1, checks)[0].passed is True

    prose = 'Here you go: {"priority": "low", "summary": "password reset"}'
    assert score_output(prose, 1, checks)[0].passed is True

    wrong = '{"priority": "high", "summary": "password reset"}'
    failed = score_output(wrong, 1, checks)[0]
    assert failed.passed is False
    assert failed.detail is not None
    assert "priority" in failed.detail

    assert score_output("not json", 1, checks)[0].detail == "response is not JSON"
