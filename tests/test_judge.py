"""Judge parsing drops ungrounded scores. No live model."""

from __future__ import annotations

import json

from prompt_eval.judge import interpret_judge_payload


OUTPUT = "Refunds are available for 30 days."


def _payload(**overrides: object) -> str:
    body: dict[str, object] = {
        "passed": True,
        "score": 1,
        "reason": "The output states the refund window.",
        "evidence": "30 days",
    }
    body.update(overrides)
    return json.dumps(body)


def test_grounded_verdict_keeps_the_models_score() -> None:
    result = interpret_judge_payload(_payload(), OUTPUT)
    assert result.status == "scored"
    assert result.passed is True
    assert result.score == 1.0
    assert result.evidence == "30 days"


def test_fail_with_empty_evidence_is_grounded() -> None:
    result = interpret_judge_payload(
        _payload(passed=False, score=0, reason="The window is missing.", evidence=""),
        OUTPUT,
    )
    assert result.status == "scored"
    assert result.passed is False
    assert result.score == 0.0
    assert result.evidence == ""


def test_pass_without_evidence_does_not_invent_a_score() -> None:
    result = interpret_judge_payload(_payload(evidence=""), OUTPUT)
    assert result.status == "ungrounded"
    assert result.score is None
    assert result.passed is None
    assert result.reason is None


def test_evidence_missing_from_output_drops_the_score() -> None:
    text = _payload(score=0.99, evidence="lifetime warranty", extra_case="made-up-case")
    result = interpret_judge_payload(text, OUTPUT)
    assert result.status == "ungrounded"
    assert result.score is None
    assert result.passed is None
    assert "0.99" not in json.dumps(result.__dict__)


def test_non_json_and_wrong_types_are_errors_without_a_score() -> None:
    bad_json = interpret_judge_payload("definitely not json", OUTPUT)
    assert bad_json.status == "error"
    assert bad_json.score is None

    string_bool = interpret_judge_payload(_payload(passed="true"), OUTPUT)
    assert string_bool.status == "error"
    assert string_bool.score is None

    bool_score = interpret_judge_payload(_payload(score=True), OUTPUT)
    assert bool_score.status == "error"
    assert bool_score.score is None

    out_of_range = interpret_judge_payload(_payload(score=1.2), OUTPUT)
    assert out_of_range.status == "error"
    assert out_of_range.score is None


def test_extra_case_list_is_ignored() -> None:
    text = _payload()
    document = json.loads(text)
    document["cases"] = [{"id": "made-up-case", "passed": True, "score": 1}]
    result = interpret_judge_payload(json.dumps(document), OUTPUT)
    assert result.status == "scored"
    assert not hasattr(result, "cases")
    assert "made-up-case" not in json.dumps(result.__dict__)
