"""Deterministic checks. These are the pass/fail source when a case has any."""

from __future__ import annotations

import re
from dataclasses import dataclass

from prompt_eval.models import CheckSpec
from prompt_eval.schema_check import explain_mismatch, parse_json_document


@dataclass(frozen=True)
class CheckResult:
    """One expectation. `passed` is computed from the model output, not supplied by the user."""

    kind: str
    expected: str
    passed: bool
    detail: str | None


def score_output(output: str, latency_ms: float, checks: CheckSpec) -> list[CheckResult]:
    """Score every configured check. Earlier failures do not skip later checks."""
    results: list[CheckResult] = []
    for substring in checks.contains:
        passed = substring in output
        results.append(
            CheckResult(
                kind="contains",
                expected=substring,
                passed=passed,
                detail=None if passed else f"missing substring {substring!r}",
            )
        )
    for substring in checks.not_contains:
        passed = substring not in output
        results.append(
            CheckResult(
                kind="not_contains",
                expected=substring,
                passed=passed,
                detail=None if passed else f"found forbidden substring {substring!r}",
            )
        )
    for pattern in checks.regex:
        passed = re.search(pattern, output) is not None
        results.append(
            CheckResult(
                kind="regex",
                expected=pattern,
                passed=passed,
                detail=None if passed else f"pattern did not match: {pattern}",
            )
        )
    if checks.json_schema is not None:
        results.append(_score_json(output, checks.json_schema))
    if checks.max_latency_ms is not None:
        limit = checks.max_latency_ms
        passed = latency_ms <= limit
        results.append(
            CheckResult(
                kind="max_latency_ms",
                expected=str(limit),
                passed=passed,
                detail=None if passed else f"latency {latency_ms:.1f} ms exceeded {limit} ms",
            )
        )
    return results


def _score_json(output: str, schema: dict[str, object]) -> CheckResult:
    try:
        document = parse_json_document(output)
    except ValueError:
        return CheckResult(
            kind="json_schema",
            expected="json_schema",
            passed=False,
            detail="response is not JSON",
        )
    reason = explain_mismatch(document, schema)
    return CheckResult(
        kind="json_schema",
        expected="json_schema",
        passed=reason is None,
        detail=reason,
    )
