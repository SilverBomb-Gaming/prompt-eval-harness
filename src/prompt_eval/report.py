"""Text and JSON reports. Every row is a case the runner actually visited."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from prompt_eval.judge import JudgeResult
from prompt_eval.scorers import CheckResult


@dataclass
class CaseReport:
    """One case that was listed (dry run) or attempted (real run)."""

    id: str
    status: str
    passed: bool | None = None
    latency_ms: float | None = None
    output: str | None = None
    error: str | None = None
    pass_source: str | None = None
    checks: list[CheckResult] = field(default_factory=list)
    judge: JudgeResult | None = None
    planned_checks: list[str] = field(default_factory=list)
    planned_judge: bool = False


@dataclass
class RunReport:
    """The whole command result. `results` never includes a case that was not selected."""

    suite_path: str
    name: str
    provider: str | None
    model: str | None
    dry_run: bool
    results: list[CaseReport]
    aborted: str | None = None

    @property
    def ok(self) -> bool:
        if self.dry_run:
            return self.aborted is None
        if self.aborted:
            return False
        return all(item.status == "scored" and item.passed is True for item in self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for item in self.results if item.status == "scored" and item.passed is True)

    @property
    def failed_count(self) -> int:
        return sum(1 for item in self.results if item.status == "scored" and item.passed is False)

    @property
    def error_count(self) -> int:
        return sum(1 for item in self.results if item.status == "error")


def to_payload(report: RunReport, results_path: str | None) -> dict[str, object]:
    """JSON document printed by --json and written to the results file."""
    if report.dry_run:
        return {
            "schema_version": 1,
            "dry_run": True,
            "suite": report.suite_path,
            "name": report.name,
            "ok": report.ok,
            "case_count": len(report.results),
            "cases": [
                {
                    "id": item.id,
                    "checks": list(item.planned_checks),
                    "judge": item.planned_judge,
                }
                for item in report.results
            ],
            "results_path": results_path,
        }
    return {
        "schema_version": 1,
        "dry_run": False,
        "suite": report.suite_path,
        "name": report.name,
        "provider": report.provider,
        "model": report.model,
        "ok": report.ok,
        "aborted": report.aborted,
        "case_count": len(report.results),
        "passed_count": report.passed_count,
        "failed_count": report.failed_count,
        "error_count": report.error_count,
        "results": [_case_payload(item) for item in report.results],
        "results_path": results_path,
    }


def render_text(report: RunReport) -> str:
    """Human report. Dry runs list ids and check names and do not claim a score."""
    if report.dry_run:
        return _render_dry_run(report)
    return _render_scored(report)


def dumps_payload(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2) + "\n"


def _render_dry_run(report: RunReport) -> str:
    lines = [f"Suite {report.name} ({len(report.results)} cases) from {report.suite_path}"]
    for item in report.results:
        checks = ", ".join(item.planned_checks) if item.planned_checks else "-"
        judge = "yes" if item.planned_judge else "no"
        lines.append(f"- {item.id}  checks: {checks}  judge: {judge}")
    lines.append("Dry run. No model calls.")
    return "\n".join(lines) + "\n"


def _render_scored(report: RunReport) -> str:
    provider = report.provider or "-"
    model = report.model or "-"
    lines = [
        f"prompt-eval  {report.name}",
        f"suite: {report.suite_path}",
        f"provider: {provider}   model: {model}",
        "",
    ]
    headers = ("ID", "RESULT", "LATENCY", "CHECKS", "JUDGE", "SOURCE")
    rows = [headers]
    for item in report.results:
        rows.append(
            (
                item.id,
                _result_label(item),
                _latency_label(item.latency_ms),
                _checks_label(item),
                _judge_label(item.judge),
                item.pass_source or "-",
            )
        )
    widths = [max(len(row[index]) for row in rows) for index in range(len(headers))]
    for row in rows:
        lines.append("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)).rstrip())
    lines.append("")
    lines.append(
        f"{len(report.results)} cases, {report.passed_count} passed, "
        f"{report.failed_count} failed, {report.error_count} errors"
    )
    detail_lines = _detail_lines(report)
    if detail_lines:
        lines.append("")
        lines.extend(detail_lines)
    return "\n".join(lines) + "\n"


def _detail_lines(report: RunReport) -> list[str]:
    lines: list[str] = []
    for item in report.results:
        notes: list[str] = []
        if item.status == "error" and item.error:
            notes.append(item.error)
        for check in item.checks:
            if not check.passed and check.detail:
                notes.append(f"{check.kind}: {check.detail}")
        if item.judge is not None and item.judge.status == "scored" and item.judge.reason:
            if item.pass_source == "checks" and item.judge.passed is not item.passed:
                verdict = "pass" if item.judge.passed else "fail"
                notes.append(f"judge {verdict} (not the pass/fail source): {item.judge.reason}")
            elif item.pass_source == "judge":
                notes.append(f"judge: {item.judge.reason}")
        if item.judge is not None and item.judge.status in {"error", "ungrounded"} and item.judge.detail:
            notes.append(f"judge {item.judge.status}: {item.judge.detail}")
        if not notes:
            continue
        lines.append(item.id)
        lines.extend(f"  {note}" for note in notes)
    return lines


def _case_payload(item: CaseReport) -> dict[str, object]:
    return {
        "id": item.id,
        "passed": item.passed,
        "status": item.status,
        "latency_ms": item.latency_ms,
        "output": item.output,
        "error": item.error,
        "pass_source": item.pass_source,
        "checks": [
            {
                "kind": check.kind,
                "expected": check.expected,
                "passed": check.passed,
                "detail": check.detail,
            }
            for check in item.checks
        ],
        "judge": _judge_payload(item.judge),
    }


def _judge_payload(judge: JudgeResult | None) -> dict[str, object] | None:
    if judge is None:
        return None
    return {
        "status": judge.status,
        "passed": judge.passed,
        "score": judge.score,
        "reason": judge.reason,
        "evidence": judge.evidence,
        "detail": judge.detail,
    }


def _result_label(item: CaseReport) -> str:
    if item.status == "error":
        return "ERROR"
    if item.passed is True:
        return "PASS"
    if item.passed is False:
        return "FAIL"
    return "-"


def _latency_label(latency_ms: float | None) -> str:
    if latency_ms is None:
        return "-"
    return f"{latency_ms:.1f} ms"


def _checks_label(item: CaseReport) -> str:
    if item.status == "error" or not item.checks:
        return "-"
    passed = sum(1 for check in item.checks if check.passed)
    return f"{passed}/{len(item.checks)}"


def _judge_label(judge: JudgeResult | None) -> str:
    if judge is None:
        return "-"
    if judge.status == "skipped" and judge.detail == "no rubric":
        return "-"
    if judge.status == "scored":
        verdict = "pass" if judge.passed else "fail"
        score = "-" if judge.score is None else f"{judge.score:.2f}"
        return f"{verdict} {score}"
    return judge.status
