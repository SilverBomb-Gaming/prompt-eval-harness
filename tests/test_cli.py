"""CLI paths with a scripted model client. No Ollama."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from prompt_eval.cli import app
from tests.support import ScriptedClient
from tests.test_suite import EXPECTED_IDS, YAML_SAMPLE

runner = CliRunner()
PASSING = [
    "Refunds are available for 30 days.",
    '{"priority":"low","summary":"password reset today"}',
    "A unit test checks a small piece of code.",
    "Good morning, Ada.",
]


def test_dry_run_lists_cases_and_does_not_build_a_client(monkeypatch) -> None:
    def boom(*args: object, **kwargs: object) -> ScriptedClient:
        raise AssertionError("dry-run built a client")

    monkeypatch.setattr("prompt_eval.cli.build_client", boom)
    result = runner.invoke(app, ["run", "--suite", str(YAML_SAMPLE), "--dry-run", "--no-save"])
    assert result.exit_code == 0, result.output
    output = result.stdout
    for case_id in EXPECTED_IDS:
        assert case_id in output
    assert "Dry run. No model calls." in output
    assert "PASS" not in output
    assert "FAIL" not in output


def test_dry_run_json_has_no_scores(monkeypatch) -> None:
    monkeypatch.setattr("prompt_eval.cli.build_client", lambda *args, **kwargs: ScriptedClient([]))
    result = runner.invoke(
        app,
        ["run", "--suite", str(YAML_SAMPLE), "--dry-run", "--json", "--no-save"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert [item["id"] for item in payload["cases"]] == EXPECTED_IDS
    assert "passed_count" not in payload
    assert "score" not in result.stdout
    assert all("passed" not in item for item in payload["cases"])


def test_run_passes_and_fails_with_the_scripted_client(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("prompt_eval.cli.build_client", lambda *args, **kwargs: ScriptedClient(PASSING))
    results = tmp_path / "results.json"
    passed = runner.invoke(
        app,
        [
            "run",
            "--suite",
            str(YAML_SAMPLE),
            "--no-judge",
            "--results",
            str(results),
        ],
    )
    assert passed.exit_code == 0, passed.output
    saved = json.loads(results.read_text(encoding="utf-8"))
    assert [row["id"] for row in saved["results"]] == EXPECTED_IDS
    assert saved["ok"] is True
    assert saved["results_path"] == str(results)
    assert "made-up-case" not in results.read_text(encoding="utf-8")

    monkeypatch.setattr(
        "prompt_eval.cli.build_client",
        lambda *args, **kwargs: ScriptedClient(
            [
                "Refunds are available for 30 days.",
                '{"priority":"low","summary":"password reset today"}',
                "Hello there.",
                "Good morning, Ada.",
            ]
        ),
    )
    failed = runner.invoke(
        app,
        ["run", "--suite", str(YAML_SAMPLE), "--no-judge", "--no-save"],
    )
    assert failed.exit_code == 1, failed.output
    assert "unit-test-plain" in failed.stdout
    assert "FAIL" in failed.stdout
    assert "made-up-case" not in failed.stdout


def test_case_filter_unknown_id_and_flag_conflict(monkeypatch) -> None:
    monkeypatch.setattr(
        "prompt_eval.cli.build_client",
        lambda *args, **kwargs: ScriptedClient(["Refunds are available for 30 days."]),
    )
    one = runner.invoke(
        app,
        ["run", "--suite", str(YAML_SAMPLE), "--case", "exact-refund", "--no-judge", "--no-save"],
    )
    assert one.exit_code == 0, one.output
    assert "exact-refund" in one.stdout
    assert "json-priority" not in one.stdout
    assert "greeting-bounds" not in one.stdout

    missing = runner.invoke(
        app,
        ["run", "--suite", str(YAML_SAMPLE), "--case", "not-a-real-case", "--dry-run", "--no-save"],
    )
    assert missing.exit_code == 2
    combined = (missing.stdout or "") + (missing.stderr or "")
    assert "not-a-real-case" in combined
    assert "PASS" not in (missing.stdout or "")

    both = runner.invoke(
        app,
        ["run", "--suite", str(YAML_SAMPLE), "--dry-run", "--results", "out.json", "--no-save"],
    )
    assert both.exit_code == 2
    assert "not both" in ((both.stdout or "") + (both.stderr or ""))


def test_provider_and_model_flags_reach_the_client(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def build(provider: str | None = None, model: str | None = None) -> ScriptedClient:
        seen["provider"] = provider
        seen["model"] = model
        return ScriptedClient(["Refunds are available for 30 days."])

    monkeypatch.setattr("prompt_eval.cli.build_client", build)
    result = runner.invoke(
        app,
        [
            "run",
            "--suite",
            str(YAML_SAMPLE),
            "--case",
            "exact-refund",
            "--no-judge",
            "--no-save",
            "--provider",
            "openai",
            "--model",
            "custom-model",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen == {"provider": "openai", "model": "custom-model"}
