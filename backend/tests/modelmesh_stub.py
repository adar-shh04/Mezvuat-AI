"""A mocked ModelMesh for tests, built from the frozen contract (`docs/contract/openapi-v1.yaml`).

Everything here is derived from the OpenAPI document, so the tests cannot drift from the contract:
- `/v1/contract` and `/v1/routing/calibration` serve the examples in the spec;
- every request the client sends is validated against the spec's request schema;
- every response the stub returns is validated against the spec's response schema before it is sent.
The base URL is a reserved test host. No test talks to a real ModelMesh.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import httpx
import respx
import yaml
from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012
from ulid import ULID

SPEC_PATH = Path(__file__).resolve().parents[2] / "docs" / "contract" / "openapi-v1.yaml"
BASE = "http://modelmesh.test"
SPEC_URI = "urn:modelmesh:openapi-v1"

ERROR_TYPES = {
    400: "invalid_request",
    401: "authentication_error",
    402: "budget_exceeded",
    403: "permission_denied",
    404: "not_found",
    409: "conflict",
    413: "request_too_large",
    422: "no_eligible_model",
    429: "rate_limited",
    500: "internal_error",
    501: "not_implemented",
    502: "upstream_error",
    503: "service_unavailable",
    504: "upstream_timeout",
}


@cache
def spec() -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    return loaded


@cache
def _registry() -> Registry:
    resource = Resource.from_contents(spec(), default_specification=DRAFT202012)
    return Registry().with_resource(SPEC_URI, resource)


@cache
def validator(component: str) -> Draft202012Validator:
    return Draft202012Validator({"$ref": f"{SPEC_URI}#/components/schemas/{component}"}, registry=_registry())


def assert_valid(component: str, instance: Any) -> None:
    errors = sorted(validator(component).iter_errors(instance), key=lambda e: list(e.path))
    assert not errors, f"{component} violates the contract: " + "; ".join(
        f"{'/'.join(map(str, e.path))}: {e.message}" for e in errors[:5]
    )


def operation(path: str, method: str) -> dict[str, Any]:
    op: dict[str, Any] = spec()["paths"][path][method]
    return op


def example(path: str, method: str = "get") -> dict[str, Any]:
    content = operation(path, method)["responses"]["200"]["content"]["application/json"]
    return copy.deepcopy(content["example"])


def chat_error_statuses() -> list[int]:
    """Every non-2xx status the spec lists for `POST /v1/chat/completions`."""
    return sorted(int(code) for code in operation("/v1/chat/completions", "post")["responses"] if code[0] != "2")


def mmr_id() -> str:
    return f"mmr_{ULID()}"


TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"


def modelmesh_block(
    *,
    request_id: str,
    client_request_id: str = "llmreq_x",
    provider: str = "groq",
    model: str = "groq/openai/gpt-oss-120b",
    cache_status: str = "miss",
    cost: float = 0.00059,
    prompt_tokens: int = 120,
    completion_tokens: int = 30,
    trace_id: str = TRACE_ID,
    structured: bool = True,
    streaming: bool = False,
) -> dict[str, Any]:
    block = {
        "request_id": request_id,
        "client_request_id": client_request_id,
        "trace_id": trace_id,
        "provider": provider,
        "model": model,
        "routing": {
            "mode": "rules_fallback",
            "policy": "quality_balanced",
            "router_version": None,
            "task_type": "general",
            "candidates_considered": 3,
            "candidates_eligible": 2,
            "predicted_quality": None,
            "predicted_quality_interval": None,
            "predicted_latency_ms": 900,
            "predicted_latency_p90_ms": 1800,
            "estimated_cost_usd": 0.0008,
            "reason": "rules policy",
            "warnings": ["quality_uncalibrated:general"],
        },
        "attempts": [
            {
                "provider": provider,
                "model": model,
                "outcome": "success",
                "latency_ms": 700,
                "http_status": 200,
                "retryable": False,
                "possibly_billed": True,
            }
        ],
        "fallback_used": False,
        "cache": {"status": cache_status, "similarity": None, "original_request_id": None, "saved_estimated_usd": None},
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": None,
            "reasoning_tokens": None,
            "source": "provider_reported",
        },
        "cost": {
            "estimated_usd": cost,
            "provider_reported_usd": None,
            "failed_attempts_estimated_usd": 0,
            "pricing_version": "2026-11-01",
        },
        "latency_ms": {"total": 760, "provider": 700, "overhead": 60, "ttft": 200 if streaming else None},
        "structured_output": {"mode": "native_strict", "validated": True, "repair_attempts": 0} if structured else None,
        "streaming": {"mode": "native"} if streaming else None,
    }
    assert_valid("ModelMeshMetadata", block)
    return block


def completion(content: str, *, request_id: str | None = None, structured: bool = True, **meta: Any) -> httpx.Response:
    rid = request_id or mmr_id()
    block = modelmesh_block(request_id=rid, structured=structured, **meta)
    body = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1794200000,
        "model": block["model"],
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": block["usage"]["prompt_tokens"],
            "completion_tokens": block["usage"]["completion_tokens"],
            "total_tokens": block["usage"]["prompt_tokens"] + block["usage"]["completion_tokens"],
        },
        "modelmesh": block,
    }
    assert_valid("ChatCompletionResponse", body)
    return httpx.Response(200, json=body, headers={"X-ModelMesh-Request-ID": rid, "X-ModelMesh-Contract": "1.0"})


def error(status: int, code: str | None = None, *, retry_after: int | None = None) -> httpx.Response:
    rid = mmr_id()
    body = {
        "error": {
            "type": ERROR_TYPES[status],
            "code": code,
            "message": "mocked",
            "param": None,
            "request_id": rid,
            "client_request_id": None,
            "retryable": status >= 500 or status == 429,
            "details": {},
        }
    }
    assert_valid("ErrorResponse", body)
    headers = {"X-ModelMesh-Request-ID": rid, "X-ModelMesh-Contract": "1.0"}
    if status == 429:
        headers["Retry-After"] = str(1 if retry_after is None else retry_after)
    elif retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return httpx.Response(status, json=body, headers=headers)


def chunk(text: str | None = None, *, finish: str | None = None) -> dict[str, Any]:
    delta: dict[str, Any] = {"content": text} if text is not None else {}
    event = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1794200000,
        "model": "groq/openai/gpt-oss-120b",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": None}],
    }
    assert_valid("ChatCompletionChunk", event)
    return event


def usage_chunk(request_id: str, **meta: Any) -> dict[str, Any]:
    block = modelmesh_block(request_id=request_id, structured=False, streaming=True, **meta)
    event = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1794200000,
        "model": block["model"],
        "choices": [],
        "usage": {
            "prompt_tokens": block["usage"]["prompt_tokens"],
            "completion_tokens": block["usage"]["completion_tokens"],
            "total_tokens": block["usage"]["prompt_tokens"] + block["usage"]["completion_tokens"],
        },
        "modelmesh": block,
    }
    assert_valid("ChatCompletionChunk", event)
    return event


def interrupted_event(request_id: str) -> dict[str, Any]:
    event = {
        "error": {
            "type": "upstream_error",
            "code": "stream_interrupted",
            "message": "upstream closed",
            "param": None,
            "request_id": request_id,
            "client_request_id": None,
            "retryable": False,
            "details": {},
        },
        "modelmesh": modelmesh_block(request_id=request_id, structured=False, streaming=True),
    }
    assert_valid("ErrorResponse", event)
    return event


def sse(events: Iterable[dict[str, Any]], *, done: bool = True, request_id: str | None = None) -> httpx.Response:
    lines = [": keep-alive"]
    lines += [f"data: {json.dumps(e)}" for e in events]
    if done:
        lines.append("data: [DONE]")
    body = ("\n\n".join(lines) + "\n\n").encode()
    headers = {"content-type": "text/event-stream", "X-ModelMesh-Contract": "1.0"}
    if request_id:
        headers["X-ModelMesh-Request-ID"] = request_id
    return httpx.Response(200, content=body, headers=headers)


def text_stream(text: str, *, request_id: str | None = None) -> httpx.Response:
    rid = request_id or mmr_id()
    words = text.split(" ")
    events = [chunk(w + (" " if i < len(words) - 1 else "")) for i, w in enumerate(words)]
    events += [chunk(finish="stop"), usage_chunk(rid)]
    return sse(events, request_id=rid)


Responder = httpx.Response | Exception | Callable[[dict[str, Any]], httpx.Response]


@dataclass
class MeshStub:
    """Routes for one test. `chat` is a queue of responses (or exceptions, or callables of the request body)."""

    router: respx.Router
    contract: dict[str, Any] = field(default_factory=lambda: example("/v1/contract"))
    calibration: dict[str, Any] = field(default_factory=lambda: example("/v1/routing/calibration"))
    chat: list[Responder] = field(default_factory=list)
    health_status: int = 200
    calibration_status: int = 200
    bodies: list[dict[str, Any]] = field(default_factory=list)
    headers: list[httpx.Headers] = field(default_factory=list)
    contract_calls: int = 0
    calibration_calls: int = 0
    health_calls: int = 0

    def __post_init__(self) -> None:
        self.router.get(f"{BASE}/v1/contract").mock(side_effect=self._contract)
        self.router.get(f"{BASE}/v1/routing/calibration").mock(side_effect=self._calibration)
        self.router.get(f"{BASE}/v1/health").mock(side_effect=self._health)
        self.chat_route = self.router.post(f"{BASE}/v1/chat/completions").mock(side_effect=self._chat)

    def _contract(self, request: httpx.Request) -> httpx.Response:
        self.contract_calls += 1
        assert "authorization" not in request.headers  # unauthenticated endpoint
        assert_valid("ContractInfo", self.contract)
        return httpx.Response(200, json=self.contract, headers={"X-ModelMesh-Contract": "1.0"})

    def _calibration(self, request: httpx.Request) -> httpx.Response:
        self.calibration_calls += 1
        if self.calibration_status != 200:
            return error(self.calibration_status)
        assert_valid("RoutingCalibration", self.calibration)
        return httpx.Response(200, json=self.calibration)

    def _health(self, request: httpx.Request) -> httpx.Response:
        self.health_calls += 1
        status = {200: "ok", 503: "down"}.get(self.health_status, "degraded")
        body = {
            "status": status,
            "contract_version": "1.0",
            "components": {
                "postgres": {"status": "ok"},
                "redis": {"status": "ok"},
                "providers": {"ok": 1, "degraded": 0, "down": 0},
            },
        }
        assert_valid("Health", body)
        return httpx.Response(self.health_status, json=body)

    def _chat(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert_valid("ChatCompletionRequest", body)
        self.bodies.append(body)
        self.headers.append(request.headers)
        if not self.chat:
            raise AssertionError("no mocked ModelMesh response left")
        item = self.chat.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item) and not isinstance(item, httpx.Response):
            return item(body)
        return item

    def calibrate(self, task_type: str, calibrated: bool = True) -> None:
        self.calibration["task_types"] = [
            {
                "task_type": task_type,
                "calibrated": calibrated,
                "acceptance_criterion": "mocked",
                "acceptance_criterion_version": "v1",
                "benchmark_manifest": None,
                "calibration_samples": 100 if calibrated else 0,
                "ece": 0.05 if calibrated else None,
                "brier": 0.1 if calibrated else None,
                "judge_human_kappa": None,
                "models": [],
            }
        ]

    def without_extension(self, path: str) -> None:
        self.contract["extensions"] = [e for e in self.contract["extensions"] if e != path]
