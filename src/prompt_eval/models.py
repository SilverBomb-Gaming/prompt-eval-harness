"""Suite documents loaded from YAML or JSON."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from prompt_eval.schema_check import assert_schema_shape

_MAX_TEXT = 100_000
_MAX_RUBRIC = 20_000
_CASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


class CheckSpec(BaseModel):
    """Deterministic expectations for one case. Every populated field is scored."""

    model_config = ConfigDict(extra="forbid")

    contains: list[str] = Field(default_factory=list)
    not_contains: list[str] = Field(default_factory=list)
    regex: list[str] = Field(default_factory=list)
    json_schema: dict[str, object] | None = None
    max_latency_ms: int | None = None

    @field_validator("contains", "not_contains", "regex", mode="before")
    @classmethod
    def _string_or_list(cls, value: object) -> object:
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("contains", "not_contains", "regex")
    @classmethod
    def _non_empty_strings(cls, value: list[str]) -> list[str]:
        for item in value:
            if not isinstance(item, str) or item == "":
                raise ValueError("must be a list of non-empty strings")
        return value

    @model_validator(mode="after")
    def _check_rules(self) -> CheckSpec:
        for pattern in self.regex:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid regex {pattern!r}: {exc}") from exc
        if self.max_latency_ms is not None and self.max_latency_ms <= 0:
            raise ValueError("max_latency_ms must be a positive integer")
        if self.json_schema is not None:
            assert_schema_shape(self.json_schema)
        return self

    def is_empty(self) -> bool:
        return (
            not self.contains
            and not self.not_contains
            and not self.regex
            and self.json_schema is None
            and self.max_latency_ms is None
        )

    def kinds(self) -> list[str]:
        """Check names in the order they are scored. Used by dry-run listings."""
        found: list[str] = []
        if self.contains:
            found.append("contains")
        if self.not_contains:
            found.append("not_contains")
        if self.regex:
            found.append("regex")
        if self.json_schema is not None:
            found.append("json_schema")
        if self.max_latency_ms is not None:
            found.append("max_latency_ms")
        return found


class JudgeSpec(BaseModel):
    """Optional rubric scored by the same model after deterministic checks."""

    model_config = ConfigDict(extra="forbid")

    rubric: str

    @field_validator("rubric")
    @classmethod
    def _rubric(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("rubric must not be empty")
        if len(stripped) > _MAX_RUBRIC:
            raise ValueError("rubric is too long")
        return stripped


class Message(BaseModel):
    """One chat message. Roles are the three the chat endpoints accept."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def _content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be empty")
        if len(value) > _MAX_TEXT:
            raise ValueError("content is too long")
        return value


class Case(BaseModel):
    """One prompt and the checks that score its model output."""

    model_config = ConfigDict(extra="forbid")

    id: str
    prompt: str | None = None
    system: str | None = None
    messages: list[Message] | None = None
    checks: CheckSpec = Field(default_factory=CheckSpec)
    judge: JudgeSpec | None = None

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        if not _CASE_ID.fullmatch(value):
            raise ValueError("id must be 1-64 characters: letters, digits, '.', '_', or '-'")
        return value

    @model_validator(mode="after")
    def _case_rules(self) -> Case:
        has_prompt = self.prompt is not None
        has_messages = self.messages is not None
        if has_prompt == has_messages:
            raise ValueError("provide exactly one of 'prompt' or 'messages'")
        if has_prompt:
            assert self.prompt is not None
            if not self.prompt.strip():
                raise ValueError("prompt must not be empty")
            if len(self.prompt) > _MAX_TEXT:
                raise ValueError("prompt is too long")
        if self.system is not None:
            if has_messages:
                raise ValueError("move system text into messages when 'messages' is set")
            if not self.system.strip():
                raise ValueError("system must not be empty")
            if len(self.system) > _MAX_TEXT:
                raise ValueError("system is too long")
        if has_messages:
            assert self.messages is not None
            if not self.messages:
                raise ValueError("messages must not be empty")
            if not any(message.role == "user" for message in self.messages):
                raise ValueError("messages must include a user message")
        if self.checks.is_empty() and self.judge is None:
            raise ValueError("add deterministic checks or a judge rubric")
        return self

    def chat_messages(self) -> list[dict[str, str]]:
        """Messages sent to the model for this case. The judge uses a different prompt."""
        if self.messages is not None:
            return [{"role": message.role, "content": message.content} for message in self.messages]
        messages: list[dict[str, str]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        assert self.prompt is not None
        messages.append({"role": "user", "content": self.prompt})
        return messages

    def task_text(self) -> str:
        """The user task, shown to the judge as context and not as model output."""
        if self.prompt is not None:
            return self.prompt
        assert self.messages is not None
        user_turns = [message.content for message in self.messages if message.role == "user"]
        return user_turns[-1]


class Suite(BaseModel):
    """A named list of cases. Ids are unique inside the file."""

    model_config = ConfigDict(extra="forbid")

    name: str
    cases: list[Case]

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("name must not be empty")
        return stripped

    @field_validator("cases")
    @classmethod
    def _cases(cls, value: list[Case]) -> list[Case]:
        if not value:
            raise ValueError("cases must not be empty")
        ids = [case.id for case in value]
        duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate case id(s): {', '.join(duplicates)}")
        return value
