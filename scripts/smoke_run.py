"""Live smoke test of the research workflow on real arXiv PDFs, without Postgres or Qdrant.

PDFs are parsed and chunked with the production code, searched with an in-memory BM25 (bm25s), and answered by
the real graph through the configured direct backend. Each citation is re-checked against the parsed text.
Nothing is written anywhere; prompts and document text are not printed except the verified quotes.

Run from backend/ (local model, no quota):
    uv run --no-sync python ../scripts/smoke_run.py --backend ollama --pdf <file.pdf> --question "..."
A Groq run (`--backend groq`) uses the free-tier quota: only public-corpus PDFs, and sparingly (decision D1, D4).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import bm25s

from mezvuat.citations import PageSpan
from mezvuat.core.config import Settings
from mezvuat.inference import build_llm_client
from mezvuat.ingestion.chunker import chunk_document
from mezvuat.ingestion.parsers.pdf import PdfParser
from mezvuat.orchestration import Budget, ResearchState, RunEvent, RunFilters, WorkflowDeps, build_graph, run_to_end
from mezvuat.retrieval.types import ChunkRecord, RankedEvidence

_WORD = re.compile(r"\S+")


class WordCounter:
    """Whitespace tokens: close enough for chunk sizing in a smoke test, and needs no model download."""

    def count(self, text: str) -> int:
        return len(_WORD.findall(text))

    def split(self, text: str, max_tokens: int) -> list[tuple[int, int]]:
        words = [m.span() for m in _WORD.finditer(text)]
        return [(words[i][0], words[min(i + max_tokens, len(words)) - 1][1]) for i in range(0, len(words), max_tokens)]


class MemoryBM25:
    def __init__(self, records: list[ChunkRecord]) -> None:
        self.records = records
        self.retriever = bm25s.BM25()
        self.retriever.index(
            bm25s.tokenize([r.text for r in records], stopwords="en", show_progress=False), show_progress=False
        )

    def search(self, workspace_id: str, query: str, k: int, filters: RunFilters) -> list[RankedEvidence]:
        tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
        ids, scores = self.retriever.retrieve(tokens, k=min(k, len(self.records)), show_progress=False)
        return [
            RankedEvidence(self.records[int(i)], rank, float(s), rank, None, None, None)
            for rank, (i, s) in enumerate(zip(ids[0], scores[0], strict=True), start=1)
            if float(s) > 0
        ]


def load(pdfs: list[Path]) -> tuple[list[ChunkRecord], dict[str, list[PageSpan]], dict[str, str]]:
    records: list[ChunkRecord] = []
    pages: dict[str, list[PageSpan]] = {}
    texts: dict[str, str] = {}
    for path in pdfs:
        parsed = PdfParser().parse(path.read_bytes(), filename=path.name)
        version_id = f"ver_{path.stem}"
        texts[version_id] = parsed.text
        pages[version_id] = [PageSpan(page=p.page_number, start=p.char_start, end=p.char_end) for p in parsed.pages]
        for draft in chunk_document(parsed, WordCounter()):
            if draft.section_kind == "references":
                continue
            records.append(
                ChunkRecord(
                    chunk_id=f"{path.stem}:{draft.ordinal}",
                    workspace_id="ws_smoke",
                    document_id=f"doc_{path.stem.split('v')[0]}",
                    document_version_id=version_id,
                    text=draft.text,
                    text_sha256=draft.text_sha256,
                    embed_text=draft.text,
                    ordinal=draft.ordinal,
                    char_start=draft.char_start,
                    char_end=draft.char_end,
                    page_start=draft.page_start,
                    page_end=draft.page_end,
                    section_path=draft.section_path,
                    section_kind=draft.section_kind,
                    document_title=parsed.title,
                    version_label=path.stem[-2:] if path.stem[-2:-1] == "v" else None,
                )
            )
    return records, pages, texts


class StaticPages:
    def __init__(self, spans: dict[str, list[PageSpan]]) -> None:
        self.spans = spans

    def page_spans(self, workspace_id: str, version_ids: Sequence[str]) -> dict[str, list[PageSpan]]:
        return {v: self.spans[v] for v in version_ids if v in self.spans}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, action="append", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--backend", choices=["ollama", "groq"], default="ollama")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--large-model", default=None)
    parser.add_argument("--policy", choices=["any", "local_only"], default=None)
    parser.add_argument("--deadline", type=float, default=600.0)
    parser.add_argument("--call-timeout", type=float, default=240.0)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--evidence-chars", type=int, default=8000)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    if args.backend == "ollama":
        settings = Settings(
            direct_public_backend="ollama",
            ollama_base_url=args.ollama_url,
            ollama_model=args.model,
            ollama_model_large=args.large_model,
            groq_api_key=None,
        )
        policy = args.policy or "local_only"
    else:
        settings = Settings(direct_public_backend="groq", ollama_base_url=None)
        policy = "any"
        if args.policy == "local_only":
            print("Groq never serves local_only data (decision D1).", file=sys.stderr)
            return 2

    records, pages, texts = load(args.pdf)
    print(f"{len(records)} chunks from {len(args.pdf)} PDF(s); backend={args.backend}; policy={policy}")
    deps = WorkflowDeps(
        llm=build_llm_client(settings),
        search=MemoryBM25(records),
        pages=StaticPages(pages),
        per_query_k=args.k,
        max_call_timeout_s=args.call_timeout,
        min_loop_time_s=args.call_timeout,
        synthesis_evidence_chars=args.evidence_chars,
    )
    initial = ResearchState(
        run_id=f"run_smoke{int(time.time())}",
        workspace_id="ws_smoke",
        question=args.question,
        data_policy=policy,
        budget=Budget(llm_calls_left=12, tokens_left=150_000, deadline=time.time() + args.deadline),
    )
    started = time.monotonic()

    def show(event: RunEvent) -> None:
        if event.event == "node":
            print(f"[{time.monotonic() - started:6.1f}s] node {event.data['node']}")
        elif event.event in ("answer_reset", "answer_replaced"):
            print(f"[{time.monotonic() - started:6.1f}s] {event.event}: {event.data.get('reason')}")

    final = run_to_end(build_graph(deps), initial, show)
    print(f"\nstatus={final.status} error={final.error}")
    for call in final.llm_calls:
        print(
            f"  {call.workflow_node:16} {call.task_type:22} {call.provider}/{call.model} {call.status:14} "
            f"{call.latency_ms:6d} ms  tokens={call.prompt_tokens}/{call.completion_tokens} {call.error_code or ''}"
        )
    for verdict in final.verification or []:
        print(f"  claim {verdict.claim_id}: {verdict.status} {verdict.problems}")
    if final.answer is None:
        return 1
    print(f"\nrendering={final.answer.rendering} flags={final.answer.flags}\n")
    print(final.answer.markdown)
    print("\nCitations re-checked against the parsed PDF text:")
    ok = True
    for citation in final.answer.citations:
        source = texts[citation.document_version_id][citation.char_start : citation.char_end]
        match = source == citation.quote
        ok &= match
        print(f"  [{citation.evidence_id}] p.{citation.page} {citation.char_start}-{citation.char_end} match={match}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
