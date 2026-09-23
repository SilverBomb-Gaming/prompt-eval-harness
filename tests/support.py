"""Shared fakes for tests that must not call a model."""

from __future__ import annotations

import json

from prompt_eval.errors import LLMError
from prompt_eval.llm import Completion


class ScriptedClient:
    """Returns queued replies. `fail_on` raises on that zero-based call index."""

    provider = "fake"
    model = "fake-model"

    def __init__(
        self,
        replies: list[str],
        *,
        latency_ms: float = 5.0,
        fail_on: int | None = None,
    ) -> None:
        self.replies = list(replies)
        self.latency_ms = latency_ms
        self.fail_on = fail_on
        self.calls: list[dict[str, object]] = []
        self._n = 0

    def complete(self, *, messages: list[dict[str, str]], json_mode: bool = False) -> Completion:
        call_index = self._n
        self._n += 1
        self.calls.append({"messages": messages, "json_mode": json_mode})
        if self.fail_on is not None and call_index == self.fail_on:
            raise LLMError("Could not reach Ollama at http://127.0.0.1:11434.")
        if not self.replies:
            raise AssertionError("ScriptedClient has no reply left")
        return Completion(text=self.replies.pop(0), latency_ms=self.latency_ms)

    def close(self) -> None:
        return None


def judge_body(
    *,
    passed: bool,
    score: float,
    reason: str,
    evidence: str,
    extra: dict[str, object] | None = None,
) -> str:
    payload: dict[str, object] = {
        "passed": passed,
        "score": score,
        "reason": reason,
        "evidence": evidence,
    }
    if extra:
        payload.update(extra)
    return json.dumps(payload)
