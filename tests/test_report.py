"""Report rows stay tied to real cases."""

from __future__ import annotations

from pathlib import Path

from prompt_eval.report import render_text, to_payload
from prompt_eval.runner import execute
from prompt_eval.suite import load_suite
from tests.support import ScriptedClient
from tests.test_suite import EXPECTED_IDS, YAML_SAMPLE


def test_scored_report_lists_each_real_case_once() -> None:
    suite = load_suite(YAML_SAMPLE)
    client = ScriptedClient(
        [
            "Refunds are available for 30 days.",
            '{"priority":"low","summary":"password reset today"}',
            "A unit test checks a small piece of code.",
            "Good morning, Ada.",
        ],
        latency_ms=12,
    )
    report = execute(
        suite=suite,
        suite_path=Path("samples/demo.yaml"),
        cases=list(suite.cases),
        client=client,
        no_judge=True,
        dry_run=False,
    )
    payload = to_payload(report, ".prompt-eval/last-results.json")
    assert payload["ok"] is True
    assert [row["id"] for row in payload["results"]] == EXPECTED_IDS
    assert payload["case_count"] == 4
    assert payload["passed_count"] == 4
    assert payload["failed_count"] == 0
    text = render_text(report)
    for case_id in EXPECTED_IDS:
        assert case_id in text
    assert "4 cases, 4 passed, 0 failed, 0 errors" in text
    assert "made-up-case" not in text
    assert "PASS" in text
