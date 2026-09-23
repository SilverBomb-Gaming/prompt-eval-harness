"""The judge prompt forbids inventing facts. No model."""

from __future__ import annotations

from prompt_eval.models import Case
from prompt_eval.prompts import GUARDRAIL_PHRASES, JUDGE_SYSTEM_PROMPT, judge_user_prompt
from prompt_eval.suite import load_suite
from tests.test_suite import YAML_SAMPLE


def test_system_prompt_contains_every_guardrail_phrase() -> None:
    for phrase in GUARDRAIL_PHRASES:
        assert phrase in JUDGE_SYSTEM_PROMPT
    assert "Do not invent facts" in JUDGE_SYSTEM_PROMPT
    assert "evidence must be copied from the model output" in JUDGE_SYSTEM_PROMPT
    assert "Do not add case ids" in JUDGE_SYSTEM_PROMPT


def test_user_prompt_is_one_case_and_does_not_include_siblings() -> None:
    suite = load_suite(YAML_SAMPLE)
    case = next(item for item in suite.cases if item.id == "exact-refund")
    sibling_ids = [item.id for item in suite.cases if item.id != case.id]
    prompt = judge_user_prompt(case, "Refunds are available for 30 days.")
    assert isinstance(case, Case)
    assert "exact-refund" in prompt
    assert "Refunds are available for 30 days." in prompt
    assert case.judge is not None
    assert case.judge.rubric in prompt
    for sibling_id in sibling_ids:
        assert sibling_id not in prompt
    assert "password reset" not in prompt
    assert "Ada" not in prompt
    assert JUDGE_SYSTEM_PROMPT not in prompt
