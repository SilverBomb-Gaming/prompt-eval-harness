"""Write the last run to a local JSON file."""

from __future__ import annotations

from pathlib import Path

from prompt_eval.report import dumps_payload

DEFAULT_RESULTS_PATH = Path(".prompt-eval") / "last-results.json"


def save_results(path: Path, payload: dict[str, object]) -> None:
    """Replace the file with this run. The parent directory is created when needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(dumps_payload(payload), encoding="utf-8")
