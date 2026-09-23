"""Errors that stop a run before a scored pass/fail exists."""


class PromptEvalError(Exception):
    """User-facing failure. The CLI prints it and exits 2."""


class SuiteError(PromptEvalError):
    """The suite file is missing, unreadable, or invalid."""


class UsageError(PromptEvalError):
    """The command flags cannot be applied to this suite."""


class LLMError(PromptEvalError):
    """The model provider rejected the configuration or the call failed."""
