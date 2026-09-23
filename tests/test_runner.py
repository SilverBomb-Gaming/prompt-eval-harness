"""Runner: deterministic checks win, and the report only lists cases that ran."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prompt_eval.errors import UsageError
from prompt_eval.report import to_payload
from prompt_eval.runner import execute
from prompt_eval.suite import load_suite, select_cases
from tests.support import ScriptedClient, judge_body
from tests.test_suite import YAML_SAMPLE


def _run(case_ids: list[str], client: ScriptedClient, *, no_judge: bool = False, dry_run: bool = False):
    suite = load_suite(YAML_SAMPLE)
    chosen = [case for case in select_cases(suite, None) if case.id in set(case_ids)]
    return execute(
        suite=suite,
        suite_path=Path("samples/demo.yaml"),
        cases=chosen,
        client=client,
        no_judge=no_judge,
        dry_run=dry_run,
    )


def test_checks_are_the_pass_fail_source_when_present() -> None:
    client = ScriptedClient(
        [
            "totally unrelated",
            judge_body(
                passed=True,
                score=1,
                reason="The output states the window.",
                evidence="totally unrelated",
            ),
        ]
    )
    report = _run(["exact-refund"], client)
    assert report.results[0].passed is False
    assert report.results[0].pass_source == "checks"
    assert report.results[0].judge is not None
    assert report.results[0].judge.passed is True
    assert report.ok is False

    client_fail_judge = ScriptedClient(
        [
            "Refunds are available for 30 days.",
            judge_body(
                passed=False,
                score=0,
                reason="The sentence is incomplete.",
                evidence="",
            ),
        ]
    )
    passed = _run(["exact-refund"], client_fail_judge)
    assert passed.results[0].passed is True
    assert passed.results[0].judge is not None
    assert passed.results[0].judge.passed is False
    text = __import__("prompt_eval.report", fromlist=["render_text"]).render_text(passed)
    assert "not the pass/fail source" in text


def test_ungrounded_judge_does_not_change_a_check_result() -> None:
    client = ScriptedClient(
        [
            "Refunds are available for 30 days.",
            judge_body(
                passed=False,
                score=0.13,
                reason="Invented a second policy.",
                evidence="lifetime warranty",
                extra={"id": "made-up-case"},
            ),
        ]
    )
    report = _run(["exact-refund"], client)
    payload = to_payload(report, None)
    encoded = json.dumps(payload)
    assert report.results[0].passed is True
    assert report.results[0].pass_source == "checks"
    assert report.results[0].judge is not None
    assert report.results[0].judge.status == "ungrounded"
    assert report.results[0].judge.score is None
    assert "made-up-case" not in encoded
    assert "0.13" not in encoded
    assert [row["id"] for row in payload["results"]] == ["exact-refund"]


def test_judge_only_case_uses_the_grounded_verdict(tmp_path: Path) -> None:
    path = tmp_path / "judge-only.yaml"
    path.write_text(
        """
name: judge-only
cases:
  - id: tone
    prompt: Say hello
    judge:
      rubric: Pass if the output says hello.
""",
        encoding="utf-8",
    )
    suite = load_suite(path)
    client = ScriptedClient(
        [
            "hello there",
            judge_body(passed=True, score=0.8, reason="It says hello.", evidence="hello"),
        ]
    )
    report = execute(
        suite=suite,
        suite_path=path,
        cases=list(suite.cases),
        client=client,
        no_judge=False,
        dry_run=False,
    )
    assert report.results[0].pass_source == "judge"
    assert report.results[0].passed is True
    assert report.results[0].checks == []
    assert client.calls[0]["json_mode"] is False
    assert client.calls[1]["json_mode"] is True
    system = client.calls[1]["messages"][0]["content"]
    assert "Do not invent facts" in system

    ungrounded = ScriptedClient(
        [
            "hello there",
            judge_body(
                passed=True,
                score=0.13,
                reason="Invented a refund policy.",
                evidence="lifetime warranty",
                extra={"cases": [{"id": "made-up-case", "score": 0.13}]},
            ),
        ]
    )
    dropped = execute(
        suite=suite,
        suite_path=path,
        cases=list(suite.cases),
        client=ungrounded,
        no_judge=False,
        dry_run=False,
    )
    payload = to_payload(dropped, None)
    encoded = json.dumps(payload)
    assert dropped.results[0].passed is False
    assert dropped.results[0].judge is not None
    assert dropped.results[0].judge.score is None
    assert "made-up-case" not in encoded
    assert "0.13" not in encoded
    assert [row["id"] for row in payload["results"]] == ["tone"]


def test_no_judge_refuses_a_case_with_nothing_else_to_score(tmp_path: Path) -> None:
    path = tmp_path / "judge-only.yaml"
    path.write_text(
        """
name: judge-only
cases:
  - id: tone
    prompt: Say hello
    judge:
      rubric: Pass if the output says hello.
""",
        encoding="utf-8",
    )
    suite = load_suite(path)
    client = ScriptedClient([])
    with pytest.raises(UsageError, match="nothing to score"):
        execute(
            suite=suite,
            suite_path=path,
            cases=list(suite.cases),
            client=client,
            no_judge=True,
            dry_run=False,
        )
    assert client.calls == []


def test_report_rows_match_attempted_cases_only() -> None:
    client = ScriptedClient(
        [
            "Refunds are available for 30 days.",
            '{"priority": "low", "summary": "password reset"}',
        ],
        fail_on=1,
    )
    suite = load_suite(YAML_SAMPLE)
    chosen = [case for case in suite.cases if case.id in {"exact-refund", "json-priority", "greeting-bounds"}]
    report = execute(
        suite=suite,
        suite_path=Path("samples/demo.yaml"),
        cases=chosen,
        client=client,
        no_judge=True,
        dry_run=False,
    )
    payload = to_payload(report, None)
    ids = [row["id"] for row in payload["results"]]
    assert ids == ["exact-refund", "json-priority"]
    assert "greeting-bounds" not in ids
    assert "unit-test-plain" not in json.dumps(payload)
    assert payload["case_count"] == 2
    assert payload["passed_count"] + payload["failed_count"] + payload["error_count"] == 2
    assert payload["passed_count"] == 1
    assert payload["error_count"] == 1
    assert payload["results"][1]["passed"] is None
    assert payload["results"][1]["checks"] == []
    assert payload["results"][1]["status"] == "error"
    assert report.aborted is not None
    assert report.ok is False


def test_dry_run_does_not_call_the_client() -> None:
    client = ScriptedClient(["should not be used"])
    suite = load_suite(YAML_SAMPLE)
    report = execute(
        suite=suite,
        suite_path=Path("samples/demo.yaml"),
        cases=list(suite.cases),
        client=client,
        no_judge=False,
        dry_run=True,
    )
    assert client.calls == []
    assert [item.id for item in report.results] == [case.id for case in suite.cases]
    assert all(item.passed is None for item in report.results)
    payload = to_payload(report, None)
    assert "passed_count" not in payload
    assert "score" not in json.dumps(payload)
    assert payload["case_count"] == 4
