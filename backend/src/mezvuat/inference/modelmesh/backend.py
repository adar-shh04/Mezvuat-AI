"""`ModelMeshBackend`: chat completions through ModelMesh's public contract 1.0 (ADR-0011).

What every request carries (only extensions listed by `GET /v1/contract`; see `negotiation.py`):
- `model: "auto"` unless the request pins one; pinned requests send `routing.fallback: "none"`.
- `routing`: the per-task policy, `timeout_ms` = the remaining deadline, `data_policy` from decision D1 (`any` for
  public-corpus workspaces, `local_only` otherwise), `required_capabilities` for structured output, and
  `min_quality` only for task types the calibration report marks calibrated and the configuration gives a value.
- `cache`: `namespace` per workspace, `context_fingerprint` over the evidence shown, `mode` `exact` for
  deterministic calls and `off` for the streamed answer or any sampled call without a seed.
- flat `metadata`: application, task_type, correlation_id (the run id), workflow_node, prompt_version, iteration.
- headers: `Authorization`, a fresh `X-Request-ID` per attempt (it is not an idempotency key), `traceparent`.

Retry rules (contract §3.1):
- network errors: one retry with jitter, then `ModelMeshUnavailableError` (the fallback trigger);
- 500/502/503/504: at most 2 retries with full-jitter backoff (or `Retry-After` when sent), then the same;
- 429: wait `Retry-After` when it fits the deadline, at most twice, else `inference_rate_limited`. Never a fallback;
- 400 `unknown_parameter`: re-read the contract once and resend; other 4xx: `ModelMeshRejectedError`, never retried;
- 422 while requiring `json_schema_strict`: resend once requiring `json_schema`.
Streams: a failure before the commit point is an HTTP error and follows the same rules. After it, an in-stream
`stream_interrupted` event or a missing `[DONE]` raises `StreamIncompleteError`: the caller discards the partial text
and retries once (the respond node does).
"""

from __future__ import annotations

import functools
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
    InferenceRateLimitedError,
    InferenceRejectedError,
    ModelMeshRejectedError,
    ModelMeshUnavailableError,
    StreamIncompleteError,
    StructuredOutputInvalidError,
)
from mezvuat.inference.http import RETRYABLE_STATUS, ErrorInfo, error_info, retry_after
from mezvuat.inference.modelmesh.keys import (
    cache_namespace,
    context_fingerprint,
    traceparent,
    valid_correlation_id,
)
from mezvuat.inference.modelmesh.negotiation import ContractInfo, ContractNegotiator
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

_MAX_TRANSPORT_RETRIES = 1
_MAX_SERVER_RETRIES = 2
_MAX_RATE_LIMIT_RETRIES = 2
_MAX_REPAIR_ERROR_CHARS = 1500
_MAX_TIMEOUT_MS = 300_000
_HTTP_SLACK_S = 2.0
_STRICT_TASKS = frozenset({"citation_verification", "evidence_synthesis"})
_COST_TASKS = frozenset({"query_classification", "query_rewriting"})

StructuredMode = Literal["json_schema_strict", "json_schema", "none"]


def default_policy(task_type: str) -> str:
    return "cost_optimized" if task_type in _COST_TASKS else "quality_balanced"


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class _Meta:
    """The parts of the `modelmesh` object Mezvuat stores (contract §2 response)."""

    request_id: str | None = None
    trace_id: str | None = None
    provider: str | None = None
    model: str | None = None
    cache_status: str | None = None
    cost_estimated_usd: float | None = None
    usage: LLMUsage | None = None

    @classmethod
    def parse(cls, block: Any) -> _Meta:
        if not isinstance(block, dict):
            return cls()
        cache = _as_dict(block.get("cache"))
        cost = _as_dict(block.get("cost"))
        usage_block = _as_dict(block.get("usage"))
        usage = None
        if usage_block:
            prompt, completion = usage_block.get("prompt_tokens"), usage_block.get("completion_tokens")
            usage = LLMUsage(
                prompt if isinstance(prompt, int) else None, completion if isinstance(completion, int) else None
            )
        estimated = cost.get("estimated_usd")

        def text(key: str) -> str | None:
            value = block.get(key)
            return value if isinstance(value, str) and value else None

        return cls(
            request_id=text("request_id"),
            trace_id=text("trace_id"),
            provider=text("provider"),
            model=text("model"),
            cache_status=cache.get("status") if isinstance(cache.get("status"), str) else None,
            cost_estimated_usd=float(estimated) if isinstance(estimated, int | float) else None,
            usage=usage,
        )


def _usage(payload: Mapping[str, Any]) -> LLMUsage | None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    return LLMUsage(prompt if isinstance(prompt, int) else None, completion if isinstance(completion, int) else None)


@dataclass
class ModelMeshBackend:
    base_url: str
    negotiator: ContractNegotiator
    namespace_secret: bytes = field(repr=False)
    api_key: str | None = field(default=None, repr=False)
    # Configured per task type; sent only when calibration says the task type is calibrated.
    min_quality: Mapping[str, float] = field(default_factory=dict)
    policies: Mapping[str, str] = field(default_factory=dict)
    http: httpx.Client = field(default_factory=lambda: httpx.Client(timeout=60.0))
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    jitter: Callable[[], float] = random.random

    # ----- request building ---------------------------------------------------------------------------------------

    def _policy(self, request: LLMRequest) -> str:
        return request.routing_policy or self.policies.get(request.task_type) or default_policy(request.task_type)

    def body(
        self,
        request: LLMRequest,
        messages: list[dict[str, str]],
        contract: ContractInfo,
        calibrated: frozenset[str],
        *,
        remaining_s: float,
        stream: bool,
        strict: bool,
    ) -> dict[str, Any]:
        """The request body, with only the extensions the contract lists (hard ones are checked at negotiation)."""
        if not valid_correlation_id(request.correlation.run_id):
            raise InferenceRejectedError(
                "the run id is not a valid ModelMesh correlation_id", code="invalid_correlation_id"
            )
        task_type: str = request.task_type
        metadata = {
            "application": "mezvuat",
            "task_type": task_type,
            "correlation_id": request.correlation.run_id,
            "workflow_node": request.correlation.workflow_node[:512],
            "prompt_version": request.correlation.prompt_version[:512],
            "iteration": str(request.correlation.iteration),
        }
        if task_type not in contract.task_types:
            metadata["task_type"] = "general"
            metadata["task_type_intended"] = task_type
        explicit = request.model is not None
        body: dict[str, Any] = {
            "model": request.model or "auto",
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "metadata": metadata,
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

        routing: dict[str, Any] = {"data_policy": request.data_policy}
        policy = self._policy(request)
        if not explicit and policy in contract.routing_policies:
            routing["policy"] = policy
        routing["timeout_ms"] = max(1, min(int(remaining_s * 1000), _MAX_TIMEOUT_MS))
        if request.response_schema is not None:
            wanted = "json_schema_strict" if strict else "json_schema"
            if wanted not in contract.capabilities and "json_schema" in contract.capabilities:
                wanted = "json_schema"
            if wanted in contract.capabilities:
                routing["required_capabilities"] = [wanted]
        quality = self.min_quality.get(task_type)
        if quality is not None and task_type in calibrated and task_type != "evaluation_judge":
            routing["min_quality"] = quality
        if explicit:
            routing["fallback"] = "none"

        cache: dict[str, Any] = {}
        sampled = request.temperature > 0 and request.seed is None
        cache["mode"] = "off" if request.task_type == "final_response" or sampled else "exact"
        if request.workspace_id is not None:
            cache["namespace"] = cache_namespace(self.namespace_secret, request.workspace_id)
            cache["context_fingerprint"] = context_fingerprint(
                request.workspace_id, request.correlation.prompt_version, request.evidence_keys
            )

        for name, group in (("routing", routing), ("cache", cache)):
            kept = {}
            for key, value in group.items():
                if contract.supports(f"{name}.{key}"):
                    kept[key] = value
                else:
                    self.negotiator.note_dropped(f"{name}.{key}")
            if kept:
                body[name] = kept
        return body

    def _headers(self, client_request_id: str, run_id: str) -> dict[str, str]:
        headers = {"X-Request-ID": client_request_id, "traceparent": traceparent(run_id)}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # ----- sending ------------------------------------------------------------------------------------------------

    def _backoff(self, attempt: int) -> float:
        return float(0.5 * (2**attempt) * self.jitter())

    def _attempt(
        self,
        body: Mapping[str, Any],
        client_request_id: str,
        started: float,
        status: Literal["ok", "invalid_output", "error"],
        *,
        http_status: int | None,
        error_code: str | None = None,
        meta: _Meta | None = None,
        mmr_id: str | None = None,
        usage: LLMUsage | None = None,
        finish_reason: str | None = None,
        model: str | None = None,
    ) -> LLMAttempt:
        meta = meta or _Meta()
        return LLMAttempt(
            client_request_id=client_request_id,
            provider=meta.provider or "modelmesh",
            model=model or meta.model or str(body.get("model", "auto")),
            provider_request_id=meta.request_id or mmr_id,
            status=status,
            http_status=http_status,
            error_code=error_code[:64] if error_code else None,
            usage=usage or meta.usage or LLMUsage(None, None, "none"),
            latency_ms=int((self.clock() - started) * 1000),
            finish_reason=finish_reason,
            backend="modelmesh",
            modelmesh_request_id=meta.request_id or mmr_id,
            modelmesh_trace_id=meta.trace_id,
            cache_status=meta.cache_status,
            cost_estimated_usd=meta.cost_estimated_usd,
        )

    def _wait(self, seconds: float, deadline: float, attempts: list[LLMAttempt], reason: str) -> None:
        if self.clock() + seconds >= deadline:
            raise ModelMeshUnavailableError("ModelMesh: no time left to retry", reason=reason, attempts=attempts)
        self.sleep(seconds)

    def _send(
        self,
        request: LLMRequest,
        build: Callable[[ContractInfo, float], dict[str, Any]],
        deadline: float,
        attempts: list[LLMAttempt],
        *,
        stream: bool,
    ) -> tuple[httpx.Response, dict[str, Any], str, float]:
        """POST with the retry rules in the module docstring. Returns the open 200 response and what was sent."""
        transport = server = limited = 0
        refreshed = False
        while True:
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise ModelMeshUnavailableError(
                    "ModelMesh: deadline exceeded", reason="modelmesh_deadline", attempts=attempts
                )
            contract = self.negotiator.contract()
            body = build(contract, remaining)
            client_request_id = new_id("llmreq")
            started = self.clock()
            try:
                http_request = self.http.build_request(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    json=body,
                    headers=self._headers(client_request_id, request.correlation.run_id),
                    # ModelMesh enforces `routing.timeout_ms`; the socket gets a little longer to see its 504.
                    timeout=httpx.Timeout(remaining + _HTTP_SLACK_S, connect=min(remaining, 10.0)),
                )
                response = self.http.send(http_request, stream=stream)
            except httpx.TransportError as exc:
                attempts.append(
                    self._attempt(
                        body, client_request_id, started, "error", http_status=None, error_code=type(exc).__name__
                    )
                )
                if transport >= _MAX_TRANSPORT_RETRIES:
                    raise ModelMeshUnavailableError(
                        f"ModelMesh unreachable: {type(exc).__name__}",
                        reason="modelmesh_unreachable",
                        attempts=attempts,
                    ) from exc
                self._wait(self._backoff(transport), deadline, attempts, "modelmesh_unreachable")
                transport += 1
                continue
            if response.status_code == 200:
                return response, body, client_request_id, started
            if stream:
                response.read()
            info = error_info(response)
            mmr_id = response.headers.get("x-modelmesh-request-id") or info.request_id
            attempts.append(
                self._attempt(
                    body,
                    client_request_id,
                    started,
                    "error",
                    http_status=response.status_code,
                    error_code=info.code or info.type or str(response.status_code),
                    mmr_id=mmr_id,
                )
            )
            if response.status_code == 429:
                wait = retry_after(response)
                wait = self._backoff(limited) if wait is None else wait
                if limited >= _MAX_RATE_LIMIT_RETRIES or self.clock() + wait >= deadline:
                    raise InferenceRateLimitedError(
                        f"ModelMesh rate limited this key (retry after {wait:.0f} s)", attempts=attempts
                    )
                self.sleep(wait)
                limited += 1
                continue
            if response.status_code in RETRYABLE_STATUS:
                reason = f"modelmesh_{response.status_code}"
                if server >= _MAX_SERVER_RETRIES:
                    raise ModelMeshUnavailableError(
                        f"ModelMesh returned {info.summary}", reason=reason, attempts=attempts
                    )
                hint = retry_after(response)
                self._wait(hint if hint is not None else self._backoff(server), deadline, attempts, reason)
                server += 1
                continue
            if response.status_code == 400 and info.code == "unknown_parameter" and not refreshed:
                # The server changed under us; read the contract again and resend only what it lists.
                refreshed = True
                self.negotiator.contract(force=True)
                continue
            raise self._rejected(info, attempts)

    @staticmethod
    def _rejected(info: ErrorInfo, attempts: list[LLMAttempt]) -> ModelMeshRejectedError:
        return ModelMeshRejectedError(
            f"ModelMesh rejected the request: {info.summary}",
            status=info.status,
            error_type=info.type,
            error_code=info.code,
            attempts=attempts,
        )

    # ----- complete -----------------------------------------------------------------------------------------------

    def complete(self, request: LLMRequest) -> LLMResult:
        deadline = self.clock() + request.timeout_s
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        attempts: list[LLMAttempt] = []
        repairs_left = request.repair_attempts if request.response_schema is not None else 0
        strict = request.response_schema is not None and request.task_type in _STRICT_TASKS
        calibrated = self.negotiator.calibrated()
        while True:
            build = functools.partial(self._build, request, list(messages), calibrated, stream=False, strict=strict)
            try:
                response, body, client_request_id, started = self._send(
                    request, build, deadline, attempts, stream=False
                )
            except ModelMeshRejectedError as exc:
                if exc.status == 422 and strict:
                    log.info(
                        "no model for json_schema_strict; retrying with json_schema", extra={"task": request.task_type}
                    )
                    strict = False
                    continue
                raise
            mmr_header = response.headers.get("x-modelmesh-request-id")
            sent = functools.partial(
                self._attempt, body, client_request_id, started, http_status=200, mmr_id=mmr_header
            )
            try:
                payload = response.json()
                choice = payload["choices"][0]
                content = choice["message"].get("content") or ""
                finish_reason = choice.get("finish_reason")
            except (json.JSONDecodeError, KeyError, IndexError, TypeError, AttributeError) as exc:
                attempts.append(sent("error", error_code="malformed"))
                raise ModelMeshUnavailableError(
                    "ModelMesh sent a malformed response", reason="modelmesh_malformed", attempts=attempts
                ) from exc
            meta = _Meta.parse(payload.get("modelmesh"))
            model = meta.model or (str(payload["model"]) if payload.get("model") else None)
            outcome = functools.partial(
                sent, meta=meta, usage=_usage(payload) or meta.usage, finish_reason=finish_reason, model=model
            )
            if request.response_schema is None:
                attempts.append(outcome("ok"))
                return self._result(content, None, model, finish_reason, attempts, "none")
            try:
                parsed = request.response_schema.model_validate_json(content)
            except ValidationError as exc:
                # ModelMesh validates against the JSON schema; Pydantic carries extra limits (lengths, counts).
                attempts.append(outcome("invalid_output", error_code="schema"))
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
            attempts.append(outcome("ok"))
            mode: StructuredMode = "json_schema_strict" if strict else "json_schema"
            return self._result(content, parsed, model, finish_reason, attempts, mode)

    def _build(
        self,
        request: LLMRequest,
        messages: list[dict[str, str]],
        calibrated: frozenset[str],
        contract: ContractInfo,
        remaining: float,
        *,
        stream: bool,
        strict: bool,
    ) -> dict[str, Any]:
        return self.body(request, messages, contract, calibrated, remaining_s=remaining, stream=stream, strict=strict)

    @staticmethod
    def _result(
        text: str,
        parsed: BaseModel | None,
        model: str | None,
        finish_reason: str | None,
        attempts: list[LLMAttempt],
        mode: StructuredMode,
    ) -> LLMResult:
        last = attempts[-1]
        return LLMResult(
            text=text,
            parsed=parsed,
            backend="modelmesh",
            provider=last.provider,
            model=model or last.model,
            finish_reason=finish_reason,
            structured_mode=mode,
            attempts=attempts,
            cost_estimated_usd=last.cost_estimated_usd,
            cache_status=last.cache_status,
        )

    # ----- stream -------------------------------------------------------------------------------------------------

    def stream(self, request: LLMRequest) -> Iterator[LLMStreamEvent]:
        if request.response_schema is not None:
            raise ValueError("the client streams only the final answer, which is free text (ADR-0008)")
        deadline = self.clock() + request.timeout_s
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        attempts: list[LLMAttempt] = []
        calibrated = self.negotiator.calibrated()

        build = functools.partial(self._build, request, messages, calibrated, stream=True, strict=False)
        response, body, client_request_id, started = self._send(request, build, deadline, attempts, stream=True)
        mmr_id = response.headers.get("x-modelmesh-request-id")
        parts: list[str] = []
        meta = _Meta()
        usage: LLMUsage | None = None
        finish_reason: str | None = None
        model: str | None = None
        done = False
        interrupted = False
        try:
            for line in response.iter_lines():
                if not line or line.startswith(":"):
                    continue  # blank separators and `: keep-alive` comments
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
                if not isinstance(event, dict):
                    continue
                if "modelmesh" in event:
                    meta = _Meta.parse(event["modelmesh"])
                if "error" in event:
                    interrupted = True
                    break
                model = str(event.get("model") or model or "") or None
                usage = _usage(event) or usage
                # The include_usage chunk has `choices: []`; never index it blindly.
                for choice in event.get("choices") or []:
                    if not isinstance(choice, dict):
                        continue
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        parts.append(delta)
                        yield StreamDelta(delta)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
        except httpx.TransportError as exc:
            attempts.append(
                self._attempt(
                    body,
                    client_request_id,
                    started,
                    "error",
                    http_status=200,
                    error_code="interrupted",
                    meta=meta,
                    mmr_id=mmr_id,
                )
            )
            raise StreamIncompleteError("ModelMesh stream interrupted", attempts=attempts) from exc
        finally:
            response.close()
        if not done:
            code = "stream_interrupted" if interrupted else "stream_incomplete"
            attempts.append(
                self._attempt(
                    body,
                    client_request_id,
                    started,
                    "error",
                    http_status=200,
                    error_code=code,
                    meta=meta,
                    mmr_id=mmr_id,
                    usage=usage,
                )
            )
            raise StreamIncompleteError(f"ModelMesh stream ended without [DONE] ({code})", attempts=attempts)
        attempts.append(
            self._attempt(
                body,
                client_request_id,
                started,
                "ok",
                http_status=200,
                meta=meta,
                mmr_id=mmr_id,
                usage=usage,
                finish_reason=finish_reason,
                model=meta.model or model,
            )
        )
        yield StreamDone(self._result("".join(parts), None, meta.model or model, finish_reason, attempts, "none"))
