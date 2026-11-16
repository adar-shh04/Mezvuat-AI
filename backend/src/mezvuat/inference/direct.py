"""Direct backends (ADR-0011): OpenAI-compatible chat completions on Groq (cloud) and Ollama (local).

No routing happens here: one fixed model per task type, set in configuration. Each backend enforces the workspace
data policy itself (decision D1): a cloud backend refuses `local_only` requests before anything is sent.

Retry rules follow the contract's fallback table, minus the fallback (there is nothing to fall back to here):
- transport errors and 500/502/503/504: retry with full-jitter backoff (base 0.5 s, factor 2, at most 2 retries),
  bounded by the request deadline, then `inference_unavailable`;
- 429: wait `Retry-After` if it fits the remaining deadline, at most twice, else `inference_rate_limited`;
- any other 4xx: `inference_rejected`, never retried.
Provider error bodies can echo the prompt, so only the error type and code are kept.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ValidationError

from mezvuat.core.ids import new_id
from mezvuat.inference.errors import (
    InferenceError,
    InferenceNotPermittedError,
    InferenceRateLimitedError,
    InferenceRejectedError,
    InferenceUnavailableError,
    StreamIncompleteError,
    StructuredOutputInvalidError,
)
from mezvuat.inference.http import RETRYABLE_STATUS, error_info, retry_after
from mezvuat.inference.schema import portable_schema
from mezvuat.inference.types import (
    LLMAttempt,
    LLMRequest,
    LLMResult,
    LLMStreamEvent,
    LLMUsage,
    StreamDelta,
    StreamDone,
)

log = logging.getLogger(__name__)
_MAX_TRANSIENT_RETRIES = 2
_MAX_RATE_LIMIT_RETRIES = 2
_MAX_REPAIR_ERROR_CHARS = 1500

Locality = Literal["cloud", "local"]


def _error_summary(response: httpx.Response) -> str:
    return error_info(response).summary


@dataclass
class OpenAICompatibleBackend:
    """One provider endpoint. `models` maps task types to model ids; `default_model` covers the rest."""

    provider: str
    base_url: str
    locality: Locality
    default_model: str
    models: Mapping[str, str] = field(default_factory=dict)
    api_key: str | None = field(default=None, repr=False)
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    max_tokens_field: str = "max_tokens"
    http: httpx.Client = field(default_factory=lambda: httpx.Client(timeout=60.0))
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    jitter: Callable[[], float] = random.random

    def model_for(self, request: LLMRequest) -> str:
        return self.models.get(request.task_type, self.default_model)

    def permits(self, request: LLMRequest) -> bool:
        return request.data_policy == "any" or self.locality == "local"

    def _check_policy(self, request: LLMRequest) -> None:
        if not self.permits(request):
            raise InferenceNotPermittedError(
                f"{self.provider} is a cloud provider and this workspace's data policy is local_only"
            )

    def _headers(self, client_request_id: str) -> dict[str, str]:
        headers = {"X-Request-ID": client_request_id}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _body(self, request: LLMRequest, messages: list[dict[str, str]], *, stream: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model_for(request),
            "messages": messages,
            "temperature": request.temperature,
            self.max_tokens_field: request.max_tokens,
            **self.extra_body,
        }
        if request.seed is not None:
            body["seed"] = request.seed
        if request.response_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.response_schema.__name__,
                    "strict": True,
                    "schema": portable_schema(request.response_schema),
                },
            }
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        return body

    def _backoff(self, attempt: int) -> float:
        return float(0.5 * (2**attempt) * self.jitter())

    def _send(
        self, request: LLMRequest, body: dict[str, Any], deadline: float, attempts: list[LLMAttempt], *, stream: bool
    ) -> tuple[httpx.Response, str, float]:
        """POST with the retry rules above. Returns the open response, its client request id and start time."""
        transient = 0
        limited = 0
        while True:
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise InferenceUnavailableError(f"{self.provider}: deadline exceeded", attempts=attempts)
            client_request_id = new_id("llmreq")
            started = self.clock()
            try:
                http_request = self.http.build_request(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=body,
                    headers=self._headers(client_request_id),
                    timeout=httpx.Timeout(remaining, connect=min(remaining, 10.0)),
                )
                response = self.http.send(http_request, stream=stream)
            except httpx.TransportError as exc:
                attempts.append(
                    self._attempt(body, client_request_id, None, started, "error", None, type(exc).__name__)
                )
                if transient >= _MAX_TRANSIENT_RETRIES:
                    raise InferenceUnavailableError(
                        f"{self.provider} unreachable: {type(exc).__name__}", attempts=attempts
                    ) from exc
                self._wait(self._backoff(transient), deadline, attempts)
                transient += 1
                continue
            if response.status_code == 200:
                return response, client_request_id, started
            if stream:
                response.read()
            summary = _error_summary(response)
            provider_id = response.headers.get("x-request-id")
            attempts.append(self._attempt(body, client_request_id, provider_id, started, "error", response, summary))
            if response.status_code == 429:
                wait = retry_after(response)
                wait = self._backoff(limited) if wait is None else wait
                if limited >= _MAX_RATE_LIMIT_RETRIES or self.clock() + wait >= deadline:
                    raise InferenceRateLimitedError(
                        f"{self.provider} rate limited (retry after {wait:.0f} s)", attempts=attempts
                    )
                self.sleep(wait)
                limited += 1
                continue
            if response.status_code in RETRYABLE_STATUS:
                if transient >= _MAX_TRANSIENT_RETRIES:
                    raise InferenceUnavailableError(f"{self.provider} returned {summary}", attempts=attempts)
                hint = retry_after(response)
                self._wait(hint if hint is not None else self._backoff(transient), deadline, attempts)
                transient += 1
                continue
            raise InferenceRejectedError(f"{self.provider} rejected the request: {summary}", attempts=attempts)

    def _wait(self, seconds: float, deadline: float, attempts: list[LLMAttempt]) -> None:
        if self.clock() + seconds >= deadline:
            raise InferenceUnavailableError(f"{self.provider}: no time left to retry", attempts=attempts)
        self.sleep(seconds)

    def _attempt(
        self,
        body: Mapping[str, Any],
        client_request_id: str,
        provider_request_id: str | None,
        started: float,
        status: Literal["ok", "invalid_output", "error"],
        response: httpx.Response | None,
        error_code: str | None,
        usage: LLMUsage | None = None,
        finish_reason: str | None = None,
    ) -> LLMAttempt:
        return LLMAttempt(
            client_request_id=client_request_id,
            provider=self.provider,
            model=str(body.get("model", self.default_model)),
            provider_request_id=provider_request_id,
            status=status,
            http_status=response.status_code if response is not None else None,
            error_code=error_code[:64] if error_code else None,
            usage=usage or LLMUsage(None, None, "none"),
            latency_ms=int((self.clock() - started) * 1000),
            finish_reason=finish_reason,
        )

    @staticmethod
    def _usage(payload: Mapping[str, Any]) -> LLMUsage:
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return LLMUsage(None, None, "none")
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
        return LLMUsage(
            int(prompt) if isinstance(prompt, int) else None,
            int(completion) if isinstance(completion, int) else None,
        )

    def complete(self, request: LLMRequest) -> LLMResult:
        self._check_policy(request)
        deadline = self.clock() + request.timeout_s
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        attempts: list[LLMAttempt] = []
        repairs_left = request.repair_attempts if request.response_schema is not None else 0
        while True:
            body = self._body(request, messages, stream=False)
            response, client_request_id, started = self._send(request, body, deadline, attempts, stream=False)
            try:
                payload = response.json()
                choice = payload["choices"][0]
                content = choice["message"].get("content") or ""
                finish_reason = choice.get("finish_reason")
            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
                attempts.append(self._attempt(body, client_request_id, None, started, "error", response, "malformed"))
                raise InferenceUnavailableError(
                    f"{self.provider} sent a malformed response", attempts=attempts
                ) from exc
            usage = self._usage(payload)
            provider_id = response.headers.get("x-request-id") or payload.get("id")
            model = str(payload.get("model") or self.model_for(request))
            if request.response_schema is None:
                attempts.append(
                    self._attempt(
                        body, client_request_id, provider_id, started, "ok", response, None, usage, finish_reason
                    )
                )
                return self._result(content, None, model, finish_reason, attempts, request)
            try:
                parsed = request.response_schema.model_validate_json(content)
            except ValidationError as exc:
                attempts.append(
                    self._attempt(
                        body,
                        client_request_id,
                        provider_id,
                        started,
                        "invalid_output",
                        response,
                        "schema",
                        usage,
                        finish_reason,
                    )
                )
                if repairs_left <= 0:
                    raise StructuredOutputInvalidError(
                        f"{request.response_schema.__name__} output failed validation", attempts=attempts
                    ) from exc
                repairs_left -= 1
                messages = [
                    *messages,
                    {"role": "assistant", "content": content[:4000]},
                    {
                        "role": "user",
                        "content": "Your previous output failed validation. Return corrected JSON only. Errors:\n"
                        + str(exc)[:_MAX_REPAIR_ERROR_CHARS],
                    },
                ]
                continue
            attempts.append(
                self._attempt(body, client_request_id, provider_id, started, "ok", response, None, usage, finish_reason)
            )
            return self._result(content, parsed, model, finish_reason, attempts, request)

    def _result(
        self,
        text: str,
        parsed: BaseModel | None,
        model: str,
        finish_reason: str | None,
        attempts: list[LLMAttempt],
        request: LLMRequest,
    ) -> LLMResult:
        return LLMResult(
            text=text,
            parsed=parsed,
            backend="direct",
            provider=self.provider,
            model=model,
            finish_reason=finish_reason,
            structured_mode="json_schema_strict" if request.response_schema is not None else "none",
            attempts=attempts,
        )

    def stream(self, request: LLMRequest) -> Iterator[LLMStreamEvent]:
        if request.response_schema is not None:
            raise ValueError("structured output cannot be streamed (Groq does not support it; ADR-0008)")
        self._check_policy(request)
        deadline = self.clock() + request.timeout_s
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        attempts: list[LLMAttempt] = []
        body = self._body(request, messages, stream=True)
        response, client_request_id, started = self._send(request, body, deadline, attempts, stream=True)
        provider_id = response.headers.get("x-request-id")
        parts: list[str] = []
        usage = LLMUsage(None, None, "none")
        finish_reason: str | None = None
        model = self.model_for(request)
        done = False
        try:
            for line in response.iter_lines():
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if "error" in event:
                    break
                model = str(event.get("model") or model)
                provider_id = provider_id or event.get("id")
                if event.get("usage"):
                    usage = self._usage(event)
                # The include_usage chunk has `choices: []`; never index it blindly.
                for choice in event.get("choices") or []:
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        parts.append(delta)
                        yield StreamDelta(delta)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
        except httpx.TransportError as exc:
            attempts.append(
                self._attempt(body, client_request_id, provider_id, started, "error", response, "interrupted")
            )
            raise StreamIncompleteError(f"{self.provider} stream interrupted", attempts=attempts) from exc
        finally:
            response.close()
        if not done:
            attempts.append(
                self._attempt(
                    body, client_request_id, provider_id, started, "error", response, "stream_incomplete", usage
                )
            )
            raise StreamIncompleteError(f"{self.provider} stream ended without [DONE]", attempts=attempts)
        attempts.append(
            self._attempt(body, client_request_id, provider_id, started, "ok", response, None, usage, finish_reason)
        )
        yield StreamDone(self._result("".join(parts), None, model, finish_reason, attempts, request))


@dataclass
class DirectClient:
    """Picks the backend the request's data policy allows (decision D1). It never widens the policy.

    `any` (public corpus): the public backend (Groq by default), else the local one. `local_only`: the local backend.
    """

    public: OpenAICompatibleBackend | None
    local: OpenAICompatibleBackend | None

    def __post_init__(self) -> None:
        if self.local is not None and self.local.locality != "local":
            raise ValueError("the private-workspace backend must run locally")

    def _select(self, request: LLMRequest) -> OpenAICompatibleBackend:
        for backend in (self.public, self.local):
            if backend is not None and backend.permits(request):
                return backend
        if request.data_policy == "local_only":
            raise InferenceNotPermittedError("private workspace and no local model backend (Ollama) is configured")
        raise InferenceNotPermittedError("no inference backend is configured (set GROQ_API_KEY or OLLAMA_BASE_URL)")

    def complete(self, request: LLMRequest) -> LLMResult:
        return self._select(request).complete(request)

    def stream(self, request: LLMRequest) -> Iterator[LLMStreamEvent]:
        return self._select(request).stream(request)


__all__ = ["DirectClient", "InferenceError", "OpenAICompatibleBackend"]
