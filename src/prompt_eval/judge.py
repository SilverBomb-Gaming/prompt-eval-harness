"""Call the judge and drop verdicts that are not grounded in the model output.

A grounded verdict has a boolean, a score from 0 to 1, a reason, and evidence
that is either empty (fail only) or an exact substring of the output.
Anything else is recorded with score null. The harness does not invent a score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from prompt_eval.errors import LLMError
from prompt_eval.models import Case
from prompt_eval.prompts import JUDGE_SYSTEM_PROMPT, judge_user_prompt
from prompt_eval.schema_check import parse_json_document


class ChatClient(Protocol):
    """The slice of the model client the judge needs."""

    def complete(self, *, messages: list[dict[str, str]], json_mode: bool = False) -> object:
        """Return an object with a `text` attribute."""


@dataclass(frozen=True)
class JudgeResult:
    """What the report stores. `score` is set only for status `scored`."""

    status: str
    passed: bool | None = None
    score: float | None = None
    reason: str | None = None
    evidence: str | None = None
    detail: str | None = None


def skipped_judge(detail: str) -> JudgeResult:
    return JudgeResult(status="skipped", detail=detail)


def score_with_judge(client: ChatClient, case: Case, output: str) -> JudgeResult:
    """Ask the model to judge one output. Transport errors become status `error`."""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": judge_user_prompt(case, output)},
    ]
    try:
        completion = client.complete(messages=messages, json_mode=True)
    except LLMError as exc:
        return JudgeResult(status="error", detail=str(exc))
    text = getattr(completion, "text", None)
    if not isinstance(text, str):
        return JudgeResult(status="error", detail="The judge client returned no text.")
    return interpret_judge_payload(text, output)


def interpret_judge_payload(text: str, output: str) -> JudgeResult:
    """Parse one judge reply. Unknown JSON keys are ignored and never become cases."""
    try:
        document = parse_json_document(text)
    except ValueError:
        return JudgeResult(status="error", detail="judge response is not JSON")
    if not isinstance(document, dict):
        return JudgeResult(status="error", detail="judge response must be a JSON object")
    passed = document.get("passed")
    score = document.get("score")
    reason = document.get("reason")
    evidence = document.get("evidence")
    if not isinstance(passed, bool):
        return JudgeResult(status="error", detail="judge field 'passed' must be a boolean")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return JudgeResult(status="error", detail="judge field 'score' must be a number from 0 to 1")
    if score < 0 or score > 1:
        return JudgeResult(status="error", detail="judge field 'score' must be a number from 0 to 1")
    if not isinstance(reason, str) or not reason.strip():
        return JudgeResult(status="error", detail="judge field 'reason' must be a non-empty string")
    if not isinstance(evidence, str):
        return JudgeResult(status="error", detail="judge field 'evidence' must be a string")
    grounded = _ground_evidence(evidence, output)
    if grounded is None:
        return JudgeResult(
            status="ungrounded",
            detail="judge evidence is not a substring of the model output",
        )
    if passed and grounded == "":
        return JudgeResult(
            status="ungrounded",
            detail="a passing judge verdict needs evidence copied from the model output",
        )
    return JudgeResult(
        status="scored",
        passed=passed,
        score=float(score),
        reason=reason.strip(),
        evidence=grounded,
        detail=None,
    )


def _ground_evidence(evidence: str, output: str) -> str | None:
    """Return evidence when it appears in the output.

    An empty string is allowed and means the judge quoted nothing.
    A non-empty string that is not in the output is rejected. The empty string
    is a substring of every string in Python, so it is handled on its own.
    """
    if evidence == "":
        return ""
    if evidence in output:
        return evidence
    stripped = evidence.strip()
    if stripped and stripped in output:
        return stripped
    if stripped == "":
        return ""
    return None
