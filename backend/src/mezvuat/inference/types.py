"""Provider-independent LLM interface (ADR-0011).

The interface is synchronous, like the rest of the request path (see `persistence/db.py`): the research graph runs
in a worker thread, and psycopg's async mode cannot run on Windows' default event loop. Streaming is an iterator.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import BaseModel

TaskType = Literal[
    "query_classification",
    "query_rewriting",
    "research_planning",
    "evidence_assessment",
    "evidence_synthesis",
    "citation_verification",
    "final_response",
    "evaluation_judge",
    "general",
]
TASK_TYPES: frozenset[str] = frozenset(TaskType.__args__)  # type: ignore[attr-defined]

# Decision D1: "any" for verified public-corpus workspaces, "local_only" for everything else.
DataPolicy = Literal["any", "local_only"]
Role = Literal["system", "user", "assistant"]
Backend = Literal["direct", "modelmesh"]
# One piece of evidence shown to the model, for the cache fingerprint: (chunk id, version id, sha256 of the text).
EvidenceKey = tuple[str, str, str]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class Correlation:
    run_id: str
    workflow_node: str
    prompt_version: str
    iteration: int = 0


@dataclass(frozen=True)
class LLMRequest:
    task_type: TaskType
    messages: list[ChatMessage]
    correlation: Correlation
    data_policy: DataPolicy
    max_tokens: int
    response_schema: type[BaseModel] | None = None
    temperature: float = 0.0
    seed: int | None = None
    timeout_s: float = 60.0
    # One repair retry with the validation error fed back (ADR-0011). The graph sets 0 when the budget is short.
    repair_attempts: int = 1
    # ModelMesh only (ADR-0011). The workspace scopes the cache namespace; the evidence keys feed the fingerprint.
    workspace_id: str | None = None
    evidence_keys: tuple[EvidenceKey, ...] = ()
    # An explicit model pin (the evaluation judge). Pinned requests never fall back to another backend or model.
    model: str | None = None
    # Overrides the per-task routing policy (the combined benchmark compares policies).
    routing_policy: str | None = None


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: int | None
    completion_tokens: int | None
    source: Literal["provider", "none"] = "provider"


@dataclass(frozen=True)
class LLMAttempt:
    """One HTTP request. Every attempt has its own `client_request_id` and becomes one `llm_call` row."""

    client_request_id: str
    provider: str
    model: str
    provider_request_id: str | None
    status: Literal["ok", "invalid_output", "error"]
    http_status: int | None
    error_code: str | None
    usage: LLMUsage
    latency_ms: int
    finish_reason: str | None
    backend: Backend = "direct"
    # ModelMesh correlation (contract §1): the `mmr_` id and the W3C trace id it reports. None on the direct path.
    modelmesh_request_id: str | None = None
    modelmesh_trace_id: str | None = None
    cache_status: str | None = None
    # ModelMesh's estimate, never billing data. None when nothing reported one.
    cost_estimated_usd: float | None = None
    # Set on direct attempts made because ModelMesh could not serve the call, e.g. `modelmesh_503`, `circuit_open`.
    fallback_reason: str | None = None


@dataclass
class LLMResult:
    text: str
    parsed: BaseModel | None
    backend: Backend
    provider: str
    model: str
    finish_reason: str | None
    structured_mode: Literal["json_schema_strict", "json_schema", "none"]
    attempts: list[LLMAttempt] = field(default_factory=list)
    # Never estimated: the direct backend reports no cost and no cache status.
    cost_estimated_usd: float | None = None
    cache_status: str | None = None
    # Why the direct backend answered instead of ModelMesh. None when no fallback happened.
    fallback_reason: str | None = None

    @property
    def usage(self) -> LLMUsage:
        prompt = [a.usage.prompt_tokens for a in self.attempts if a.usage.prompt_tokens is not None]
        completion = [a.usage.completion_tokens for a in self.attempts if a.usage.completion_tokens is not None]
        if not prompt and not completion:
            return LLMUsage(None, None, "none")
        return LLMUsage(sum(prompt), sum(completion))


@dataclass(frozen=True)
class StreamDelta:
    text: str


@dataclass(frozen=True)
class StreamDone:
    result: LLMResult


LLMStreamEvent = StreamDelta | StreamDone


class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> LLMResult: ...

    def stream(self, request: LLMRequest) -> Iterator[LLMStreamEvent]:
        """Yield text deltas, then exactly one `StreamDone`. Raises `StreamIncompleteError` without `[DONE]`."""
        ...
