"""Typed inference errors (ADR-0011). `code` values are stable and surface in run results and the API."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mezvuat.core.errors import MezvuatError

if TYPE_CHECKING:
    from mezvuat.inference.types import LLMAttempt


class InferenceError(MezvuatError):
    code = "inference_failed"
    http_status = 502

    def __init__(self, message: str, *, code: str | None = None, attempts: list[LLMAttempt] | None = None) -> None:
        super().__init__(message, code=code)
        # The HTTP attempts made before the failure, so that every request is still recorded in `llm_call`.
        self.attempts: list[LLMAttempt] = list(attempts or [])


class StructuredOutputInvalidError(InferenceError):
    """The model output failed schema validation, including after the repair retry."""

    code = "structured_output_invalid"


class InferenceUnavailableError(InferenceError):
    """Transport errors or 5xx after retries. Fatal for a run: the graph goes to `fail`."""

    code = "inference_unavailable"
    http_status = 503


class InferenceNotPermittedError(InferenceError):
    """No backend is allowed to see this workspace's data (decision D1). Nothing was sent."""

    code = "inference_not_permitted"
    http_status = 403


class InferenceRateLimitedError(InferenceError):
    """A 429 whose Retry-After does not fit the remaining deadline. Never a fallback trigger."""

    code = "inference_rate_limited"
    http_status = 429


class InferenceRejectedError(InferenceError):
    """A 4xx other than 429: configuration, policy or budget. Never retried, never a fallback trigger."""

    code = "inference_rejected"


class StreamIncompleteError(InferenceError):
    """The stream ended without `[DONE]`. The partial output must be discarded."""

    code = "stream_incomplete"


FATAL_INFERENCE_CODES = frozenset(
    {InferenceUnavailableError.code, InferenceNotPermittedError.code, InferenceRejectedError.code}
)
