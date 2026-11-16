"""Identifiers Mezvuat sends to ModelMesh, in the exact formats the contract fixes (§1, §2, §2.3).

- `cache.namespace`: `ws_` + 32 hex of HMAC-SHA256(server secret, workspace id). The workspace id never leaves.
- `cache.context_fingerprint`: `sha256:` + 64 lowercase hex over canonical JSON of the workspace, the prompt version
  and the sorted evidence keys, so reordering evidence keeps the key and any change in a chunk's text changes it.
- `metadata.correlation_id`: the research run id, checked against `[A-Za-z0-9_:.-]{1,128}`.
- `traceparent`: W3C Trace Context. The current OpenTelemetry span when one is active (Langfuse, M6), otherwise a
  trace id derived from the run id, so every call of one run shares a trace id that ModelMesh echoes back.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
from collections.abc import Iterable

from mezvuat.inference.types import EvidenceKey

CORRELATION_ID = re.compile(r"^[A-Za-z0-9_:.-]{1,128}$")
NAMESPACE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
FINGERPRINT = re.compile(r"^sha256:[0-9a-f]{64}$")
TRACE_ID = re.compile(r"^[0-9a-f]{32}$")
METADATA_KEY = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
FINGERPRINT_VERSION = 1


def cache_namespace(secret: bytes, workspace_id: str) -> str:
    if not secret:
        raise ValueError("the namespace secret is empty")
    digest = hmac.new(secret, workspace_id.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"ws_{digest[:32]}"


def evidence_key(chunk_id: str, version_id: str, text: str) -> EvidenceKey:
    return (chunk_id, version_id, hashlib.sha256(text.encode("utf-8")).hexdigest())


def context_fingerprint(workspace_id: str, prompt_version: str, evidence: Iterable[EvidenceKey]) -> str:
    canonical = json.dumps(
        {
            "v": FINGERPRINT_VERSION,
            "workspace_id": workspace_id,
            "prompt_version": prompt_version,
            "evidence": sorted({tuple(key) for key in evidence}),
        },
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=True,
    )
    return "sha256:" + hashlib.sha256(canonical.encode("ascii")).hexdigest()


def valid_correlation_id(value: str) -> bool:
    return CORRELATION_ID.match(value) is not None


def _otel_trace_id() -> str | None:
    """The active OpenTelemetry span's trace id, if the SDK is installed and a valid span is current."""
    try:
        from opentelemetry import trace
    except ImportError:
        return None
    context = trace.get_current_span().get_span_context()
    if not context.is_valid:
        return None
    return format(context.trace_id, "032x")


def run_trace_id(run_id: str) -> str:
    trace_id = hashlib.sha256(f"mezvuat-trace:{run_id}".encode()).hexdigest()[:32]
    # An all-zero trace id is invalid in W3C Trace Context; the chance is 2^-128, but keep the format total.
    return trace_id if trace_id.strip("0") else "1".rjust(32, "0")


def traceparent(run_id: str) -> str:
    trace_id = _otel_trace_id() or run_trace_id(run_id)
    span_id = secrets.token_hex(8)
    while not span_id.strip("0"):
        span_id = secrets.token_hex(8)
    return f"00-{trace_id}-{span_id}-01"
