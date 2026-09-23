"""Suite loading. No model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prompt_eval.errors import SuiteError
from prompt_eval.suite import load_suite, select_cases

ROOT = Path(__file__).resolve().parents[1]
YAML_SAMPLE = ROOT / "samples" / "demo.yaml"
JSON_SAMPLE = ROOT / "samples" / "demo.json"
EXPECTED_IDS = ["exact-refund", "json-priority", "unit-test-plain", "greeting-bounds"]


def test_sample_yaml_and_json_match() -> None:
    yaml_suite = load_suite(YAML_SAMPLE)
    json_suite = load_suite(JSON_SAMPLE)
    assert yaml_suite.name == "demo"
    assert [case.id for case in yaml_suite.cases] == EXPECTED_IDS
    assert yaml_suite.model_dump() == json_suite.model_dump()
    kinds = {case.id: case.checks.kinds() for case in yaml_suite.cases}
    assert kinds["exact-refund"] == ["contains", "not_contains", "regex", "max_latency_ms"]
    assert kinds["json-priority"] == ["contains", "json_schema"]
    assert kinds["greeting-bounds"] == ["contains", "not_contains", "regex"]
    assert yaml_suite.cases[3].messages is not None
    assert yaml_suite.cases[0].prompt is not None


def test_select_one_case_and_reject_unknown() -> None:
    suite = load_suite(YAML_SAMPLE)
    selected = select_cases(suite, "json-priority")
    assert [case.id for case in selected] == ["json-priority"]
    with pytest.raises(SuiteError, match="Unknown case 'not-a-real-case'"):
        select_cases(suite, "not-a-real-case")


def test_rejects_bad_suites(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(SuiteError, match="not found"):
        load_suite(missing)

    text_file = tmp_path / "notes.txt"
    text_file.write_text("name: demo\n", encoding="utf-8")
    with pytest.raises(SuiteError, match=".yaml"):
        load_suite(text_file)

    both = tmp_path / "both.yaml"
    both.write_text(
        """
name: bad
cases:
  - id: mixed
    prompt: hello
    messages:
      - role: user
        content: hello
    checks:
      contains: hello
""",
        encoding="utf-8",
    )
    with pytest.raises(SuiteError, match="prompt"):
        load_suite(both)

    dupes = tmp_path / "dupes.json"
    dupes.write_text(
        json.dumps(
            {
                "name": "bad",
                "cases": [
                    {"id": "same", "prompt": "one", "checks": {"contains": ["one"]}},
                    {"id": "same", "prompt": "two", "checks": {"contains": ["two"]}},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SuiteError, match="duplicate case id"):
        load_suite(dupes)

    regex = tmp_path / "regex.yaml"
    regex.write_text(
        """
name: bad
cases:
  - id: broken
    prompt: hello
    checks:
      regex:
        - "("
""",
        encoding="utf-8",
    )
    with pytest.raises(SuiteError, match="invalid regex"):
        load_suite(regex)

    empty_score = tmp_path / "empty.yaml"
    empty_score.write_text(
        """
name: bad
cases:
  - id: bare
    prompt: hello
""",
        encoding="utf-8",
    )
    with pytest.raises(SuiteError, match="deterministic checks"):
        load_suite(empty_score)

    unknown_key = tmp_path / "extra.yaml"
    unknown_key.write_text(
        """
name: bad
cases:
  - id: extra
    prompt: hello
    checks:
      contains: hello
    owner: someone
""",
        encoding="utf-8",
    )
    with pytest.raises(SuiteError, match="owner"):
        load_suite(unknown_key)


def test_string_contains_and_light_schema_keyword(tmp_path: Path) -> None:
    single = tmp_path / "single.yaml"
    single.write_text(
        """
name: short
cases:
  - id: one
    prompt: Say hello
    checks:
      contains: hello
""",
        encoding="utf-8",
    )
    suite = load_suite(single)
    assert suite.cases[0].checks.contains == ["hello"]

    schema = tmp_path / "schema.yaml"
    schema.write_text(
        """
name: bad
cases:
  - id: one
    prompt: Give JSON
    checks:
      json_schema:
        type: object
        minLength: 1
""",
        encoding="utf-8",
    )
    with pytest.raises(SuiteError, match="unsupported keyword"):
        load_suite(schema)
