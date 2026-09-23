"""Run selected cases. Rows are appended only after a case is listed or attempted."""

from __future__ import annotations

from pathlib import Path

from prompt_eval.errors import LLMError, UsageError
from prompt_eval.judge import JudgeResult, score_with_judge, skipped_judge
from prompt_eval.llm import LLMClient
from prompt_eval.models import Case, Suite
from prompt_eval.report import CaseReport, RunReport
from prompt_eval.scorers import score_output


def execute(
    *,
    suite: Suite,
    suite_path: Path,
    cases: list[Case],
    client: LLMClient | None,
    no_judge: bool,
    dry_run: bool,
) -> RunReport:
    """Score `cases` in order. A transport error stops the loop and omits later cases."""
    if no_judge and not dry_run:
        _reject_unscorable(cases)
    report = RunReport(
        suite_path=str(suite_path),
        name=suite.name,
        provider=None if client is None else client.provider,
        model=None if client is None else client.model,
        dry_run=dry_run,
        results=[],
    )
    if dry_run:
        for case in cases:
            report.results.append(
                CaseReport(
                    id=case.id,
                    status="planned",
                    planned_checks=case.checks.kinds(),
                    planned_judge=case.judge is not None,
                )
            )
        return report
    if client is None:
        raise UsageError("A model client is required unless this is a dry run.")
    for case in cases:
        try:
            report.results.append(_run_case(client, case, no_judge=no_judge))
        except LLMError as exc:
            report.results.append(
                CaseReport(
                    id=case.id,
                    status="error",
                    passed=None,
                    error=str(exc),
                )
            )
            report.aborted = str(exc)
            break
    return report


def _reject_unscorable(cases: list[Case]) -> None:
    missing = [case.id for case in cases if case.checks.is_empty()]
    if missing:
        listed = ", ".join(missing)
        raise UsageError(
            f"Case {listed} has no deterministic checks. "
            "Refusing --no-judge because there would be nothing to score."
        )


def _run_case(client: LLMClient, case: Case, *, no_judge: bool) -> CaseReport:
    completion = client.complete(messages=case.chat_messages(), json_mode=False)
    output = completion.text
    latency_ms = completion.latency_ms
    has_checks = not case.checks.is_empty()
    checks = score_output(output, latency_ms, case.checks) if has_checks else []
    judge = _judge_case(client, case, output, no_judge=no_judge)
    if has_checks:
        passed = all(check.passed for check in checks)
        pass_source = "checks"
    else:
        passed = judge.status == "scored" and judge.passed is True
        pass_source = "judge"
    return CaseReport(
        id=case.id,
        status="scored",
        passed=passed,
        latency_ms=latency_ms,
        output=output,
        error=None,
        pass_source=pass_source,
        checks=checks,
        judge=judge,
    )


def _judge_case(client: LLMClient, case: Case, output: str, *, no_judge: bool) -> JudgeResult | None:
    if case.judge is None:
        return skipped_judge("no rubric") if not case.checks.is_empty() else None
    if no_judge:
        return skipped_judge("disabled with --no-judge")
    return score_with_judge(client, case, output)
