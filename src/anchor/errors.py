"""The single exception hierarchy for anchor.

Every foreseeable failure in this repo raises one of these — never a bare
``Exception``, never a silent ``None``. ``http_status_for`` is the one place a
status code is decided, so the API layer never has to grow an if/elif ladder.
"""

from __future__ import annotations


class AnchorError(Exception):
    """Base class for every error raised by anchor."""


class ConfigError(AnchorError):
    """Invalid or contradictory configuration."""


class ProviderNotConfigured(AnchorError):
    """A provider was requested but its credentials/settings are missing."""


class ProviderError(AnchorError):
    """A remote provider call failed after retries."""


class IngestError(AnchorError):
    """A document could not be loaded, chunked or indexed."""


class IndexMismatchError(AnchorError):
    """The persisted vector index does not match the configured embedder."""


class StoreError(AnchorError):
    """A storage-layer operation failed."""


class RetrievalError(AnchorError):
    """Retrieval could not be completed."""


class NoEvidenceError(AnchorError):
    """Generation was attempted with an empty context pack (never allowed)."""


class ToolError(AnchorError):
    """An agent tool failed or was called with invalid arguments."""


class BudgetExceeded(AnchorError):
    """An agent exhausted its step, token or wall-clock budget."""


class PolicyViolation(AnchorError):
    """The request was refused by the input policy."""


class EvalError(AnchorError):
    """The evaluation harness could not run, or a golden label is unresolvable."""


HTTP_STATUS_BY_ERROR: dict[type[AnchorError], int] = {
    ConfigError: 500,
    ProviderNotConfigured: 503,
    ProviderError: 502,
    IngestError: 400,
    IndexMismatchError: 500,
    StoreError: 500,
    RetrievalError: 500,
    NoEvidenceError: 500,
    ToolError: 400,
    BudgetExceeded: 429,
    PolicyViolation: 400,
    EvalError: 500,
}


def http_status_for(exc: BaseException) -> int:
    """Map an exception to an HTTP status code (500 for anything unknown)."""
    for cls, status in HTTP_STATUS_BY_ERROR.items():
        if isinstance(exc, cls):
            return status
    return 500
