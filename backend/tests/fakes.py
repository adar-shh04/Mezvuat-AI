"""Test doubles for the research graph. Never used outside tests."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from mezvuat.citations import PageSpan
from mezvuat.core.errors import RetrievalUnavailableError
from mezvuat.core.ids import new_id
from mezvuat.inference import (
    InferenceError,
    LLMAttempt,
    LLMRequest,
    LLMResult,
    LLMStreamEvent,
    LLMUsage,
    StreamDelta,
    StreamDone,
    StreamIncompleteError,
)
from mezvuat.orchestration import LLMCallRecord, RunFilters
from mezvuat.retrieval.types import ChunkRecord, RankedEvidence

_WORD = re.compile(r"[a-z0-9]+")


def attempt(model: str = "fake-model") -> LLMAttempt:
    return LLMAttempt(
        client_request_id=new_id("llmreq"),
        provider="fake",
        model=model,
        provider_request_id=None,
        status="ok",
        http_status=200,
        error_code=None,
        usage=LLMUsage(100, 20),
        latency_ms=1,
        finish_reason="stop",
    )


@dataclass
class ScriptedLLM:
    """Returns scripted outputs per task type, in order. A script item may be a model, a dict, a string or an
    exception instance (raised). `final_response` items are strings, streamed word by word."""

    script: dict[str, list[Any]]
    requests: list[LLMRequest] = field(default_factory=list)
    default: dict[str, Any] = field(default_factory=dict)

    def _next(self, request: LLMRequest) -> Any:
        self.requests.append(request)
        queue = self.script.get(request.task_type)
        if queue:
            item = queue.pop(0)
        elif request.task_type in self.default:
            item = self.default[request.task_type]
        else:
            raise AssertionError(f"no scripted output left for {request.task_type}")
        # A callable builds its output from the request, e.g. to quote the evidence the prompt actually showed.
        return item(request) if callable(item) and not isinstance(item, type) else item

    def complete(self, request: LLMRequest) -> LLMResult:
        item = self._next(request)
        if isinstance(item, Exception):
            if isinstance(item, InferenceError) and not item.attempts:
                failed = attempt()
                item.attempts = [
                    LLMAttempt(**{**failed.__dict__, "status": "error", "error_code": item.code, "http_status": 503})
                ]
            raise item
        assert request.response_schema is not None
        parsed = item if isinstance(item, BaseModel) else request.response_schema.model_validate(item)
        parsed = request.response_schema.model_validate(parsed.model_dump())
        return LLMResult(
            text=parsed.model_dump_json(),
            parsed=parsed,
            backend="direct",
            provider="fake",
            model="fake-model",
            finish_reason="stop",
            structured_mode="json_schema_strict",
            attempts=[attempt()],
        )

    def stream(self, request: LLMRequest) -> Iterator[LLMStreamEvent]:
        item = self._next(request)
        if isinstance(item, StreamIncompleteError):
            yield StreamDelta("partial ")
            item.attempts = [attempt()]
            raise item
        if isinstance(item, Exception):
            raise item
        text = str(item)
        for word in re.findall(r"\S+\s*", text):
            yield StreamDelta(word)
        yield StreamDone(
            LLMResult(
                text=text,
                parsed=None,
                backend="direct",
                provider="fake",
                model="fake-model",
                finish_reason="stop",
                structured_mode="none",
                attempts=[attempt()],
            )
        )

    def calls(self, task_type: str) -> int:
        return sum(1 for r in self.requests if r.task_type == task_type)


def chunk(
    chunk_id: str,
    text: str,
    *,
    workspace_id: str = "ws_1",
    doc: str = "doc_1",
    version: str = "ver_1",
    char_start: int = 0,
    page_start: int | None = 1,
    page_end: int | None = 1,
    title: str = "A paper",
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        workspace_id=workspace_id,
        document_id=doc,
        document_version_id=version,
        text=text,
        text_sha256=chunk_id,
        embed_text=text,
        ordinal=0,
        char_start=char_start,
        char_end=char_start + len(text),
        page_start=page_start,
        page_end=page_end,
        section_path=("1 Introduction",),
        section_kind="body",
        is_latest_version=True,
        document_title=title,
        version_label="v1",
    )


@dataclass
class KeywordSearch:
    """Ranks chunks by query-term overlap. `fail` makes every search raise like a down Qdrant."""

    chunks: list[ChunkRecord]
    fail: bool = False
    queries: list[str] = field(default_factory=list)

    def search(self, workspace_id: str, query: str, k: int, filters: RunFilters) -> list[RankedEvidence]:
        self.queries.append(query)
        if self.fail:
            raise RetrievalUnavailableError("vector store unreachable")
        terms = set(_WORD.findall(query.lower()))
        scored = []
        for record in self.chunks:
            if record.workspace_id != workspace_id:
                continue
            overlap = len(terms & set(_WORD.findall(record.text.lower())))
            if overlap:
                scored.append((overlap, record))
        scored.sort(key=lambda pair: (-pair[0], pair[1].chunk_id))
        return [
            RankedEvidence(
                chunk=record,
                rank=rank,
                score=float(score),
                lexical_rank=rank,
                dense_rank=None,
                fusion_score=None,
                rerank_score=None,
            )
            for rank, (score, record) in enumerate(scored[:k], start=1)
        ]


@dataclass
class StaticPages:
    spans: dict[str, list[PageSpan]]

    def page_spans(self, workspace_id: str, version_ids: Sequence[str]) -> dict[str, list[PageSpan]]:
        return {v: self.spans[v] for v in version_ids if v in self.spans}


@dataclass
class ListRecorder:
    rows: list[LLMCallRecord] = field(default_factory=list)
    by_run: dict[str, list[LLMCallRecord]] = field(default_factory=lambda: defaultdict(list))

    def record(self, records: Sequence[LLMCallRecord]) -> None:
        self.rows.extend(records)


_BLOCK = re.compile(r'<evidence id="(E\d+)" boundary="[0-9a-f]+">\n[^\n]*\n\n(.*?)\n</evidence boundary=', re.DOTALL)


def shown_evidence(request: LLMRequest) -> dict[str, str]:
    """Evidence id -> text, as shown in the request's user message."""
    return dict(_BLOCK.findall(request.messages[-1].content))


def id_for(request: LLMRequest, phrase: str) -> str:
    for evidence_id, text in shown_evidence(request).items():
        if phrase in text:
            return evidence_id
    raise AssertionError(f"no evidence block contains {phrase!r}")
