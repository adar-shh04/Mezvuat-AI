"""`mezvuat-dataset`: build the domain retrieval dataset and review it.

corpus    chunk the demo corpus (latest version per paper) into the local cache
generate  generate queries for val and test (and optionally train) with Groq
pool      pool candidate passages from the four retrieval systems
prelabel  LLM pre-labels for every pooled passage
review    the product owner's labelling tool (see docs/LABELLING_GUIDE.md)
status    progress and workload
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mezvuat.core.config import get_settings
from mezvuat.core.logging import configure_logging
from mezvuat.evaluation import domain
from mezvuat.evaluation.domain import DomainChunk, GeneratedQuery

log = logging.getLogger("mezvuat.dataset")

REPO = Path(__file__).resolve().parents[4]
DATASET_DIR = REPO / "data" / "domain" / "v1"
MANIFEST = REPO / "data" / "manifests" / "demo_corpus.json"
SPLITS = REPO / "data" / "manifests" / "demo_corpus_splits.json"
SEED = 13
BLIND_FRACTION = 0.2
TARGETS = {"test": 60, "val": 40}


def cache_dir() -> Path:
    return Path(os.environ.get("MEZVUAT_DATA_ROOT", REPO / "data")) / "domain" / "v1"


def corpus_path() -> Path:
    return cache_dir() / "corpus.jsonl"


def read_manifest() -> dict[str, Any]:
    path = DATASET_DIR / "manifest.json"
    return (
        json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"dataset_version": domain.DATASET_VERSION}
    )


def write_manifest(data: dict[str, Any]) -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "manifest.json").write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8", newline="\n")


def ensure_corpus() -> dict[str, DomainChunk]:
    path = corpus_path()
    if not path.exists():
        cmd_corpus(argparse.Namespace())
    chunks = domain.load_corpus(path)
    expected = read_manifest().get("corpus", {}).get("digest")
    if expected and domain.corpus_digest(chunks.values()) != expected:
        raise SystemExit(
            "The local corpus does not match the dataset manifest. Rebuild it with `mezvuat-dataset corpus`."
        )
    return chunks


def cmd_corpus(args: argparse.Namespace) -> None:
    from mezvuat.ml.models import HfTokenCounter
    from mezvuat.ml.registry import get_embedder_spec

    spec = get_embedder_spec(get_settings().embedding_model_id)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))["assignment"]
    pdf_dir = Path(os.environ.get("MEZVUAT_DATA_ROOT", REPO / "data")) / "corpus" / "arxiv"
    chunks = domain.build_corpus(manifest, splits, pdf_dir, HfTokenCounter(spec.model_id, spec.revision))
    domain.write_jsonl(corpus_path(), (c.to_json() for c in chunks))
    meta = read_manifest()
    meta["corpus"] = {
        "digest": domain.corpus_digest(chunks),
        "n_chunks": len(chunks),
        "n_papers": len({c.arxiv_id for c in chunks}),
        "versions": "latest downloaded version of each paper",
        "chunker": "structure-v1",
        "tokenizer": f"{spec.model_id}@{spec.revision}",
        "splits_file_sha256": domain.sha256_text(SPLITS.read_text(encoding="utf-8")),
    }
    write_manifest(meta)
    log.info("corpus built", extra={"chunks": len(chunks)})


def _groq_client() -> Any:
    from mezvuat.inference.groq import GroqClient

    return GroqClient.from_settings(get_settings())


def _record_usage(meta: dict[str, Any], stage: str, client: Any, started: float) -> None:
    usage = client.usage
    meta.setdefault("usage", {})[stage] = {
        "model": client.model,
        "calls": usage.calls,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "retries_429": usage.retries_429,
        "seconds_waiting_for_rate_limits": round(usage.seconds_waiting, 1),
        "wall_clock_seconds": round(time.monotonic() - started, 1),
    }


def cmd_generate(args: argparse.Namespace) -> None:
    chunks = ensure_corpus()
    corpus = list(chunks.values())
    client = _groq_client()
    progress_path = cache_dir() / "generation_progress.jsonl"
    done = {row["source_chunk_id"]: row for row in domain.read_jsonl(progress_path)}
    targets = dict(TARGETS)
    if args.train:
        targets["train"] = args.train
    started = time.monotonic()
    with progress_path.open("a", encoding="utf-8", newline="\n") as progress:
        for split, target in targets.items():
            for index, chunk in enumerate(domain.sample_sources(corpus, split, target, seed=SEED)):
                if chunk.chunk_id in done:
                    continue
                query_type = domain.QUERY_TYPES[index % len(domain.QUERY_TYPES)]
                prompt = domain.generation_prompt(chunk, query_type)
                row: dict[str, Any] | None = None
                for attempt in range(2):
                    seed = SEED * 1000 + index + attempt * 100_000
                    result = client.structured(
                        [{"role": "system", "content": domain.GENERATION_SYSTEM}, {"role": "user", "content": prompt}],
                        schema_name="generated_query",
                        schema=domain.GENERATION_SCHEMA,
                        temperature=0.7,
                        seed=seed,
                        max_tokens=1200,
                        estimated_prompt_tokens=chunk.token_count + 400,
                    )
                    question = " ".join(str(result.data["question"]).split())
                    run = domain.longest_shared_run(question, chunk.text)
                    if question and run <= domain.MAX_SHARED_RUN and domain.is_self_contained(question):
                        row = {
                            "split": split,
                            "text": question,
                            "query_type": query_type,
                            "source_chunk_id": chunk.chunk_id,
                            "arxiv_id": chunk.arxiv_id,
                            "answer_quote": str(result.data["answer_quote"]),
                            "longest_shared_run": run,
                            "generation": {
                                "model": result.model,
                                "prompt_version": domain.GENERATION_PROMPT_VERSION,
                                "prompt_sha256": domain.sha256_text(domain.GENERATION_SYSTEM + prompt),
                                "temperature": 0.7,
                                "seed": seed,
                                "reasoning_effort": "low",
                                "attempt": attempt + 1,
                                "provider_request_id": result.request_id,
                                "prompt_tokens": result.prompt_tokens,
                                "completion_tokens": result.completion_tokens,
                            },
                        }
                        break
                if row is None:
                    row = {
                        "split": split,
                        "source_chunk_id": chunk.chunk_id,
                        "rejected": "copied_too_much_or_not_self_contained",
                    }
                progress.write(json.dumps(row, ensure_ascii=False) + "\n")
                progress.flush()
                done[chunk.chunk_id] = row
            log.info("split generated", extra={"split": split, "calls": client.usage.calls})
    _finalise_queries(list(done.values()))
    meta = read_manifest()
    _record_usage(meta, "generate", client, started)
    meta["generation"] = {
        "prompt_version": domain.GENERATION_PROMPT_VERSION,
        "system_prompt": domain.GENERATION_SYSTEM,
        "user_prompt_template": domain.generation_prompt(_template_chunk(), "factoid"),
        "type_hints": domain._TYPE_HINTS,
        "temperature": 0.7,
        "seed_rule": "13000 + source index within split; +100000 on the single retry",
        "max_shared_run": domain.MAX_SHARED_RUN,
        "targets": targets,
        "sampling": "round-robin across the split's families; eligible chunks are body or abstract, >= 80 tokens",
    }
    write_manifest(meta)


def _template_chunk() -> DomainChunk:
    return DomainChunk("", "", "", "", "{title}", ("{section}",), "body", None, None, 0, "{passage}", "")


def _finalise_queries(rows: list[dict[str, Any]]) -> None:
    from mezvuat.ml.models import SentenceTransformerEmbedder

    accepted = [r for r in rows if "text" in r]
    queries: list[GeneratedQuery] = []
    counters: dict[str, int] = {}
    for row in sorted(accepted, key=lambda r: (r["split"], r["source_chunk_id"])):
        counters[row["split"]] = counters.get(row["split"], 0) + 1
        queries.append(
            GeneratedQuery(
                query_id=f"dq_{row['split']}_{counters[row['split']]:04d}",
                split=row["split"],
                text=row["text"],
                query_type=row["query_type"],
                source_chunk_id=row["source_chunk_id"],
                arxiv_id=row["arxiv_id"],
                answer_quote=row["answer_quote"],
                longest_shared_run=row["longest_shared_run"],
                generation=row["generation"],
            )
        )
    embedder = SentenceTransformerEmbedder.from_model_id(get_settings().embedding_model_id)
    embeddings = embedder.embed_queries([q.text for q in queries])
    kept, dropped = domain.dedupe_queries(queries, embeddings)
    rejected = [r for r in rows if "rejected" in r]
    domain.write_jsonl(DATASET_DIR / "queries.jsonl", (q.__dict__ for q in kept))
    domain.write_jsonl(DATASET_DIR / "dropped_queries.jsonl", [*dropped, *rejected])


def _load_queries() -> list[dict[str, Any]]:
    return domain.read_jsonl(DATASET_DIR / "queries.jsonl")


def cmd_pool(args: argparse.Namespace) -> None:
    from mezvuat.ml.models import CrossEncoderReranker, SentenceTransformerEmbedder
    from mezvuat.retrieval.lexical import Bm25Index
    from mezvuat.retrieval.pipeline import RetrievalConfig, RetrievalPipeline
    from mezvuat.retrieval.types import ChunkPayload

    chunks = ensure_corpus()
    ids = sorted(chunks)
    store = domain.CorpusStore(chunks)
    payloads = [
        ChunkPayload(c, chunks[c].arxiv_id, chunks[c].arxiv_id, True, None, chunks[c].section_kind) for c in ids
    ]
    lexical_index = Bm25Index.build(payloads, [chunks[c].text for c in ids])
    settings = get_settings()
    embedder = SentenceTransformerEmbedder.from_model_id(settings.embedding_model_id)
    vectors = embedder.embed_documents([store.get_many("domain", [c])[c].embed_text for c in ids])
    dense = domain.ExactDenseIndex(ids, vectors, [chunks[c].section_kind for c in ids])
    reranker = CrossEncoderReranker.from_model_id("BAAI/bge-reranker-base")

    class Lexical:
        def search(self, workspace_id: str, query: str, k: int, filters: Any) -> Any:
            return lexical_index.search(query, k, filters)

    pipeline = RetrievalPipeline(Lexical(), dense, embedder, store, reranker=reranker)
    configs = {
        "bm25": RetrievalConfig(mode="bm25"),
        "dense": RetrievalConfig(mode="dense"),
        "hybrid_rrf": RetrievalConfig(mode="hybrid_rrf"),
        "hybrid_rrf_rerank": RetrievalConfig(mode="hybrid_rrf_rerank", rerank_depth=50),
    }
    rows = []
    for query in _load_queries():
        if query["split"] not in TARGETS:
            continue
        rankings = {
            name: [e.chunk.chunk_id for e in pipeline.retrieve("domain", query["text"], k=10, config=config).evidence]
            for name, config in configs.items()
        }
        pool = domain.merge_pool(query["source_chunk_id"], rankings)
        for entry in pool:
            entry["text_sha256"] = chunks[entry["chunk_id"]].text_sha256
        rows.append({"query_id": query["query_id"], "split": query["split"], "candidates": pool})
    domain.write_jsonl(DATASET_DIR / "pools.jsonl", rows)
    meta = read_manifest()
    meta["pooling"] = {
        "systems": {name: str(config) for name, config in configs.items()},
        "embedder": settings.embedding_model_id,
        "reranker": "BAAI/bge-reranker-base",
        "per_system_depth": 10,
        "pool_size": domain.POOL_SIZE,
        "rule": "source passage, then round-robin by rank across systems until the pool is full",
        "dense_search": "exact cosine over the corpus (offline pooling only)",
    }
    write_manifest(meta)
    log.info("pools written", extra={"queries": len(rows)})


def cmd_prelabel(args: argparse.Namespace) -> None:
    import random

    chunks = ensure_corpus()
    queries = {q["query_id"]: q for q in _load_queries()}
    out_path = DATASET_DIR / "prelabels.jsonl"
    existing = domain.read_jsonl(out_path)
    done = {row["query_id"] for row in existing}
    client = _groq_client()
    started = time.monotonic()
    try:
        with out_path.open("a", encoding="utf-8", newline="\n") as handle:
            for pool in sorted(domain.read_jsonl(DATASET_DIR / "pools.jsonl"), key=_val_first):
                if pool["query_id"] in done:
                    continue
                query = queries[pool["query_id"]]
                candidates = [c["chunk_id"] for c in pool["candidates"]]
                # Shuffle so the source passage is not always first; the order is recorded.
                order = list(candidates)
                random.Random(f"order:{pool['query_id']}").shuffle(order)  # noqa: S311
                pid_for = {chunk_id: f"P{n}" for n, chunk_id in enumerate(order, start=1)}
                passage_ids = [pid_for[c] for c in order]
                prompt = domain.label_prompt(query["text"], [(pid_for[c], chunks[c]) for c in order])
                labels = None
                for attempt in range(2):
                    result = client.structured(
                        [{"role": "system", "content": domain.LABEL_SYSTEM}, {"role": "user", "content": prompt}],
                        schema_name="relevance_labels",
                        schema=domain.label_schema(passage_ids),
                        temperature=0.0,
                        seed=SEED + attempt,
                        max_tokens=2500,
                        estimated_prompt_tokens=sum(chunks[c].token_count for c in order) + 600,
                    )
                    labels = domain.validate_labels(result.data, passage_ids)
                    if labels is not None:
                        break
                meta_row = {
                    "model": result.model,
                    "prompt_version": domain.LABEL_PROMPT_VERSION,
                    "prompt_sha256": domain.sha256_text(domain.LABEL_SYSTEM + prompt),
                    "temperature": 0.0,
                    "seed": SEED + attempt,
                    "reasoning_effort": "low",
                    "presented_order": [pid_for[c] for c in order],
                    "provider_request_id": result.request_id,
                }
                for chunk_id in candidates:
                    label = labels[pid_for[chunk_id]] if labels else None
                    handle.write(
                        json.dumps(
                            {
                                "query_id": pool["query_id"],
                                "split": pool["split"],
                                "chunk_id": chunk_id,
                                "passage_id": pid_for[chunk_id],
                                "grade": label["grade"] if label else None,
                                "reason": label["reason"] if label else None,
                                "labeller": meta_row,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                handle.flush()
    finally:
        # Usage is recorded even when the run stops on a quota error, so token counts are never lost.
        meta = read_manifest()
        runs = meta.setdefault("usage", {}).setdefault("prelabel_runs", [])
        scratch: dict[str, Any] = {}
        _record_usage(scratch, "run", client, started)
        runs.append({**scratch["usage"]["run"], "finished_at": _now()})
        write_manifest(meta)
    meta = read_manifest()
    meta["prelabel"] = {
        "prompt_version": domain.LABEL_PROMPT_VERSION,
        "system_prompt": domain.LABEL_SYSTEM,
        "grade_definitions": domain.GRADE_DEFINITIONS,
        "temperature": 0.0,
        "batching": "one call per query, all pooled passages in a shuffled, recorded order",
    }
    meta["review"] = {"blind_fraction": BLIND_FRACTION, "blind_seed": SEED, "blind_unit": "query"}
    write_manifest(meta)


def _val_first(pool: dict[str, Any]) -> tuple[bool, str]:
    return (pool["split"] != "val", pool["query_id"])


def cmd_review(args: argparse.Namespace) -> None:
    from mezvuat.evaluation.review import run_review

    run_review(args.split, reviewer=args.reviewer, chunks=ensure_corpus(), dataset_dir=DATASET_DIR, seed=SEED)


def cmd_status(args: argparse.Namespace) -> None:
    from mezvuat.evaluation.review import status_report

    print(status_report(DATASET_DIR, seed=SEED))


def main() -> None:
    configure_logging(get_settings().mezvuat_log_level)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(prog="mezvuat-dataset", description=__doc__)
    sub = parser.add_subparsers(required=True, dest="command")
    sub.add_parser("corpus").set_defaults(func=cmd_corpus)
    gen = sub.add_parser("generate")
    gen.add_argument("--train", type=int, default=0, help="also generate this many train queries")
    gen.set_defaults(func=cmd_generate)
    sub.add_parser("pool").set_defaults(func=cmd_pool)
    sub.add_parser("prelabel").set_defaults(func=cmd_prelabel)
    review = sub.add_parser("review")
    review.add_argument("split", choices=["val", "test"])
    review.add_argument("--reviewer", default="product-owner")
    review.set_defaults(func=cmd_review)
    sub.add_parser("status").set_defaults(func=cmd_status)
    args = parser.parse_args()
    args.func(args)


def _now() -> str:
    return datetime.now(UTC).isoformat()


if __name__ == "__main__":
    main()
