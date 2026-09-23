"""Optional .env loading. Existing variables win."""

from __future__ import annotations

from pathlib import Path

from prompt_eval.dotenv import load_dotenv


def test_dotenv_does_not_override_the_environment(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# comment",
                "PROMPT_EVAL_PROVIDER=openai",
                "OLLAMA_MODEL='custom'",
                "EMPTY=",
                "SKIP",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PROMPT_EVAL_PROVIDER", "ollama")
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    load_dotenv(env_file)
    assert __import__("os").environ["PROMPT_EVAL_PROVIDER"] == "ollama"
    assert __import__("os").environ["OLLAMA_MODEL"] == "custom"


def test_missing_dotenv_is_fine(tmp_path: Path) -> None:
    load_dotenv(tmp_path / "missing.env")
