"""Run tracing with Langfuse (ADR-0012, decision D3). Optional: without keys, or without the SDK, nothing is traced.

- One Langfuse trace per research run, with a trace id derived from the run id (`Langfuse.create_trace_id(seed=)`),
  so it is known before the run starts and stored in every `llm_call` row (`langfuse_trace_id`).
- LangGraph's node and LLM spans come from Langfuse's LangChain callback handler.
- **Masking happens inside the process, before anything is exported** (D3): every string in a span's input, output
  or metadata is replaced by its length and a short hash, except identifiers (`run_…`, `mmr_…`, `E3`) and short
  enum-like values. `LANGFUSE_CAPTURE_CONTENT=true` turns masking off; it is off by default.
- Tracing never breaks a run: set-up errors fall back to no tracing, callback errors are swallowed by LangChain,
  and exports happen on a background thread.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Protocol

log = logging.getLogger(__name__)

# Kept unmasked: ids and short, code-like values (node names, task types, statuses, model ids).
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:/@+-]{1,64}$")
_MAX_KEPT = 64


def _masked(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:12]
    return f"<masked {len(text)} chars sha256:{digest}>"


def mask_content(*, data: Any, **_: Any) -> Any:
    """Langfuse `mask` function: keep structure, numbers, booleans and identifiers; mask every other string.

    Whitespace is the tell: prompts, questions and document text always contain spaces, identifiers never do."""
    if isinstance(data, str):
        return data if len(data) <= _MAX_KEPT and _IDENTIFIER.match(data) else _masked(data)
    if isinstance(data, Mapping):
        return {str(k): mask_content(data=v) for k, v in data.items()}
    if isinstance(data, list | tuple | set | frozenset):
        return [mask_content(data=v) for v in data]
    if data is None or isinstance(data, bool | int | float):
        return data
    # Pydantic models, messages and other objects: mask their string form.
    return _masked(str(data))


class RunTracer(Protocol):
    def trace_id(self, run_id: str) -> str | None: ...

    def run(self, run_id: str, *, metadata: Mapping[str, str]) -> Any:
        """Context manager yielding the LangChain callbacks for this run (an empty list when not tracing)."""
        ...

    def score(self, run_id: str, name: str, value: float, *, comment: str | None = None) -> None: ...


class NoopTracer:
    def trace_id(self, run_id: str) -> str | None:
        return None

    @contextmanager
    def run(self, run_id: str, *, metadata: Mapping[str, str]) -> Iterator[list[Any]]:
        yield []

    def score(self, run_id: str, name: str, value: float, *, comment: str | None = None) -> None:
        return None


class LangfuseTracer:
    def __init__(self, client: Any, public_key: str | None = None) -> None:
        self._client = client
        self._public_key = public_key

    def trace_id(self, run_id: str) -> str | None:
        from langfuse import Langfuse

        trace_id: str = Langfuse.create_trace_id(seed=run_id)
        return trace_id

    @contextmanager
    def run(self, run_id: str, *, metadata: Mapping[str, str]) -> Iterator[list[Any]]:
        try:
            from langfuse.langchain import CallbackHandler

            trace_id = self.trace_id(run_id)
            observation = self._client.start_as_current_observation(
                name="research_run",
                as_type="span",
                trace_context={"trace_id": trace_id},
                metadata={"run_id": run_id, **metadata},
            )
            span = observation.__enter__()
            handler = CallbackHandler(public_key=self._public_key)
        except Exception:
            log.warning("Langfuse tracing unavailable for this run", exc_info=True)
            yield []
            return
        try:
            yield [handler]
        finally:
            try:
                span.update(output={"run_id": run_id})
                observation.__exit__(None, None, None)
            except Exception:
                log.warning("Langfuse span could not be closed", exc_info=True)

    def score(self, run_id: str, name: str, value: float, *, comment: str | None = None) -> None:
        try:
            self._client.create_score(
                name=name, value=value, trace_id=self.trace_id(run_id), data_type="NUMERIC", comment=comment
            )
        except Exception:
            log.warning("Langfuse score not sent", exc_info=True)

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception:
            log.warning("Langfuse flush failed", exc_info=True)


def build_tracer(
    *,
    public_key: str | None,
    secret_key: str | None,
    host: str | None,
    capture_content: bool = False,
    **client_options: Any,
) -> RunTracer:
    """A Langfuse tracer when both keys are set and the SDK is installed; otherwise a no-op."""
    if not public_key or not secret_key:
        return NoopTracer()
    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=host,
            mask=None if capture_content else mask_content,
            **client_options,
        )
    except Exception:
        log.warning("Langfuse is configured but could not start; tracing is off", exc_info=True)
        return NoopTracer()
    return LangfuseTracer(client, public_key)
