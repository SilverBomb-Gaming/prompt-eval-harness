"""Load a suite file and select cases."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import ValidationError

from prompt_eval.errors import SuiteError
from prompt_eval.models import Case, Suite

_MAX_BYTES = 1_000_000
_SUFFIXES = {".yaml", ".yml", ".json"}


def load_suite(path: Path) -> Suite:
    """Read UTF-8 YAML or JSON. Unknown keys and invalid checks fail here."""
    suite_path = Path(path)
    if not suite_path.is_file():
        raise SuiteError(f"Suite file not found: {suite_path}")
    if suite_path.suffix.lower() not in _SUFFIXES:
        raise SuiteError("Suite file must end in .yaml, .yml, or .json.")
    try:
        raw = suite_path.read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise SuiteError(f"Suite file is not UTF-8: {suite_path}") from exc
    if len(raw.encode("utf-8")) > _MAX_BYTES:
        raise SuiteError("Suite file is larger than 1000000 bytes.")
    data = _parse(raw, suite_path)
    if not isinstance(data, dict):
        raise SuiteError("Suite file must be a mapping with 'name' and 'cases'.")
    try:
        return Suite.model_validate(data)
    except ValidationError as exc:
        raise SuiteError(_format_validation(exc)) from exc


def select_cases(suite: Suite, case_id: str | None) -> list[Case]:
    """Return suite order, or the one case named by --case. Unknown ids are an error."""
    if case_id is None:
        return list(suite.cases)
    matches = [case for case in suite.cases if case.id == case_id]
    if not matches:
        known = ", ".join(case.id for case in suite.cases)
        raise SuiteError(f"Unknown case {case_id!r}. Cases in this suite: {known}")
    return matches


def _parse(raw: str, path: Path) -> object:
    try:
        if path.suffix.lower() == ".json":
            return json.loads(raw)
        return yaml.safe_load(raw)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise SuiteError(f"Could not parse {path.name}: {exc}") from exc


def _format_validation(exc: ValidationError) -> str:
    lines: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "suite"
        message = error["msg"]
        prefix = "Value error, "
        if message.startswith(prefix):
            message = message[len(prefix) :]
        lines.append(f"{location}: {message}")
    return "Invalid suite:\n" + "\n".join(lines)
