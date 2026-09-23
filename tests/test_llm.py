"""HTTP client construction. Transport is in-process. No Ollama."""

from __future__ import annotations

import json

import httpx
import pytest

from prompt_eval.errors import LLMError
from prompt_eval.llm import OllamaClient, OpenAICompatibleClient, build_client

ENV_KEYS = (
    "PROMPT_EVAL_PROVIDER",
    "PROMPT_EVAL_TIMEOUT",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "OLLAMA_NUM_CTX",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_default_provider_stays_ollama_when_openai_url_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = build_client()
    try:
        assert client.provider == "ollama"
        assert client.model == "llama3.2"
    finally:
        client.close()


def test_openai_host_requires_a_key_and_a_local_server_does_not(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROMPT_EVAL_PROVIDER", "openai")
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        build_client()
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1")
    client = build_client(model="local-model")
    try:
        assert isinstance(client, OpenAICompatibleClient)
        assert client.api_key == ""
        assert client.model == "local-model"
    finally:
        client.close()


def test_bad_provider_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(LLMError, match="ollama' or 'openai"):
        build_client(provider="anthropic")
    monkeypatch.setenv("PROMPT_EVAL_TIMEOUT", "nope")
    with pytest.raises(LLMError, match="PROMPT_EVAL_TIMEOUT"):
        build_client()
    monkeypatch.setenv("PROMPT_EVAL_TIMEOUT", "120")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "10")
    with pytest.raises(LLMError, match="OLLAMA_NUM_CTX"):
        build_client()


def test_ollama_json_mode_is_opt_in_and_empty_text_is_kept() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content.decode()))
        if len(seen) == 1:
            return httpx.Response(200, json={"message": {"content": ""}})
        return httpx.Response(200, json={"message": {"content": "{\"ok\": true}"}})

    client = OllamaClient(
        base_url="http://127.0.0.1:11434",
        model="llama3.2",
        timeout=5,
        num_ctx=8192,
        http=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    plain = client.complete(messages=[{"role": "user", "content": "hi"}], json_mode=False)
    judged = client.complete(
        messages=[{"role": "system", "content": "rules"}, {"role": "user", "content": "score"}],
        json_mode=True,
    )
    client.close()
    assert plain.text == ""
    assert judged.text == "{\"ok\": true}"
    assert "format" not in seen[0]
    assert seen[1]["format"] == "json"
    assert seen[0]["options"]["temperature"] == 0
    assert seen[0]["stream"] is False


def test_ollama_404_mentions_pull() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="model missing")

    client = OllamaClient(
        base_url="http://127.0.0.1:11434",
        model="llama3.2",
        timeout=5,
        num_ctx=8192,
        http=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(LLMError, match="ollama pull llama3.2"):
        client.complete(messages=[{"role": "user", "content": "hi"}])
    client.close()


def test_openai_response_format_is_only_sent_for_the_judge() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content.decode()))
        assert request.url.path.endswith("/chat/completions")
        assert request.headers["authorization"] == "Bearer sk-test"
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = OpenAICompatibleClient(
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        model="gpt-4o-mini",
        timeout=5,
        http=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    client.complete(messages=[{"role": "user", "content": "hi"}], json_mode=False)
    client.complete(messages=[{"role": "user", "content": "score"}], json_mode=True)
    client.close()
    assert "response_format" not in seen[0]
    assert seen[1]["response_format"] == {"type": "json_object"}
    assert seen[0]["temperature"] == 0


def test_connect_error_is_a_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = OllamaClient(
        base_url="http://127.0.0.1:11434",
        model="llama3.2",
        timeout=5,
        num_ctx=8192,
        http=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(LLMError, match="Could not reach Ollama"):
        client.complete(messages=[{"role": "user", "content": "hi"}])
    client.close()
