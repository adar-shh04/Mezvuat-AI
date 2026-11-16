"""`ModelMeshClient`: ModelMesh first, the direct backends as fallback (ADR-0011, contract §3.1, decision D1).

Fallback happens only when ModelMesh cannot serve the call: `ModelMeshUnavailableError` (network errors, 500/502/503/504
after retries, an unreadable contract) or an open circuit breaker. It never happens on 429 or any other 4xx, never
for a pinned model (the evaluation judge), and never mid-stream: a stream that started on ModelMesh stays there.

The fallback is a `DirectClient`, which applies the same data policy as ModelMesh did: `any` may use Groq, `local_only`
only Ollama. If no direct backend is allowed, the call fails with `inference_not_permitted`; the policy is never
widened. Direct attempts made in fallback carry `fallback_reason`, so `llm_call` rows and the UI badge show it.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterator
from dataclasses import dataclass

from mezvuat.inference.direct import DirectClient
from mezvuat.inference.errors import (
    InferenceError,
    InferenceNotPermittedError,
    ModelMeshUnavailableError,
)
from mezvuat.inference.modelmesh.backend import ModelMeshBackend
from mezvuat.inference.modelmesh.breaker import CircuitBreaker
from mezvuat.inference.types import LLMAttempt, LLMRequest, LLMResult, LLMStreamEvent, StreamDone

log = logging.getLogger(__name__)


def _tag(attempts: list[LLMAttempt], reason: str) -> list[LLMAttempt]:
    return [dataclasses.replace(a, fallback_reason=reason) for a in attempts]


@dataclass
class ModelMeshClient:
    backend: ModelMeshBackend
    breaker: CircuitBreaker
    # None for INFERENCE_MODE=modelmesh: failures surface as `inference_unavailable`.
    fallback: DirectClient | None = None

    def negotiate(self) -> None:
        """Start-up check. Raises `InferenceContractMismatchError` (a hard failure) and logs unreachability."""
        try:
            self.backend.negotiator.negotiate()
        except ModelMeshUnavailableError as exc:
            log.warning(
                "ModelMesh unreachable at start-up; calls will use the fallback rules", extra={"reason": exc.reason}
            )

    def _may_fall_back(self, request: LLMRequest) -> bool:
        return self.fallback is not None and request.model is None

    def _open(self, request: LLMRequest) -> ModelMeshUnavailableError:
        return ModelMeshUnavailableError("ModelMesh circuit breaker is open", reason="circuit_open")

    def _fallback_error(self, exc: InferenceError, before: list[LLMAttempt], reason: str) -> InferenceError:
        exc.attempts = [*before, *_tag(exc.attempts, reason)]
        if isinstance(exc, InferenceNotPermittedError):
            exc.message = f"ModelMesh could not serve the call ({reason}) and {exc.message}"
        return exc

    # ----- complete -----------------------------------------------------------------------------------------------

    def complete(self, request: LLMRequest) -> LLMResult:
        if not self.breaker.allow():
            return self._complete_fallback(request, self._open(request))
        try:
            result = self.backend.complete(request)
        except ModelMeshUnavailableError as exc:
            self.breaker.failure()
            return self._complete_fallback(request, exc)
        except InferenceError:
            # A 4xx, 429 or invalid output: ModelMesh answered, so it is up.
            self.breaker.success()
            raise
        self.breaker.success()
        return result

    def _complete_fallback(self, request: LLMRequest, cause: ModelMeshUnavailableError) -> LLMResult:
        if not self._may_fall_back(request) or self.fallback is None:
            raise cause
        log.warning("ModelMesh unavailable; using the direct fallback", extra={"reason": cause.reason})
        try:
            result = self.fallback.complete(request)
        except InferenceError as exc:
            raise self._fallback_error(exc, cause.attempts, cause.reason) from cause
        result.attempts = [*cause.attempts, *_tag(result.attempts, cause.reason)]
        result.fallback_reason = cause.reason
        return result

    # ----- stream -------------------------------------------------------------------------------------------------

    def stream(self, request: LLMRequest) -> Iterator[LLMStreamEvent]:
        if not self.breaker.allow():
            yield from self._stream_fallback(request, self._open(request))
            return
        events = self.backend.stream(request)
        try:
            # The HTTP request happens on the first event: failures before the commit point surface here.
            first = next(events)
        except StopIteration:
            self.breaker.success()
            return
        except ModelMeshUnavailableError as exc:
            self.breaker.failure()
            yield from self._stream_fallback(request, exc)
            return
        except InferenceError:
            self.breaker.success()
            raise
        self.breaker.success()
        yield first
        # After the commit point there is no backend switch: interruptions propagate to the caller's retry.
        yield from events

    def _stream_fallback(self, request: LLMRequest, cause: ModelMeshUnavailableError) -> Iterator[LLMStreamEvent]:
        if not self._may_fall_back(request) or self.fallback is None:
            raise cause
        log.warning("ModelMesh unavailable; streaming from the direct fallback", extra={"reason": cause.reason})
        try:
            for event in self.fallback.stream(request):
                if isinstance(event, StreamDone):
                    result = event.result
                    result.attempts = [*cause.attempts, *_tag(result.attempts, cause.reason)]
                    result.fallback_reason = cause.reason
                yield event
        except InferenceError as exc:
            raise self._fallback_error(exc, cause.attempts, cause.reason) from cause
