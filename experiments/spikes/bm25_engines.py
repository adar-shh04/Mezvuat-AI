"""ADR-0004 spike (decision D6): compare lexical engines on retrieval quality and operational cost.

Engines: bm25s (the production Bm25Index), ParadeDB pg_search (AGPL), and Qdrant's native BM25 text index.
Tuning data only: the SciFact dev split carved from train, never the test split.

Prerequisites: a throwaway ParadeDB container on 127.0.0.1:15434 and the compose Qdrant. Usage (from backend/):
  uv run --no-sync python ../experiments/spikes/bm25_engines.py --out ../docs/eval/bm25_engines.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import psycopg
from qdrant_client import QdrantClient, models

from mezvuat.evaluation.beir import BeirDataset, download_dataset, load_dataset, split_dev_from_train
from mezvuat.evaluation.metrics import DEFAULT_METRICS, bootstrap_ci, paired_comparison, per_query_metric
from mezvuat.retrieval.lexical import Bm25Index
from mezvuat.retrieval.types import ChunkPayload, Filters

DSN = "host=127.0.0.1 port=15434 dbname=spike user=spike password=spike-only"
QDRANT_URL = "http://127.0.0.1:16333"
COLLECTION = "spike__bm25__scifact"
TOP_K = 100
Search = Callable[[str], list[str]]


def bm25s_engine(dataset: BeirDataset) -> tuple[Search, float]:
    doc_ids = sorted(dataset.corpus)
    payloads = [ChunkPayload(d, d, d, True, None, "body") for d in doc_ids]
    start = time.perf_counter()
    index = Bm25Index.build(payloads, [dataset.corpus[d].full_text for d in doc_ids])
    build = time.perf_counter() - start
    none = Filters(exclude_section_kinds=frozenset())

    def search(query: str) -> list[str]:
        return [hit.chunk_id for hit in index.search(query, TOP_K, none)]

    return search, build


def paradedb_engine(dataset: BeirDataset) -> tuple[Search, float]:
    connection = psycopg.connect(DSN, autocommit=True)
    connection.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
    connection.execute("DROP TABLE IF EXISTS docs")
    connection.execute("CREATE TABLE docs (id text PRIMARY KEY, body text)")
    with connection.cursor() as cursor, cursor.copy("COPY docs (id, body) FROM STDIN") as copy:
        for doc_id, doc in dataset.corpus.items():
            copy.write_row((doc_id, doc.full_text))
    start = time.perf_counter()
    connection.execute(
        "CREATE INDEX docs_bm25 ON docs USING bm25 "
        "(id, (body::pdb.simple('stemmer=english', 'stopwords_language=english'))) "
        "WITH (key_field='id')"
    )
    build = time.perf_counter() - start

    def search(query: str) -> list[str]:
        rows = connection.execute(
            "SELECT id, pdb.score(id) AS s FROM docs WHERE body ||| %s ORDER BY s DESC, id LIMIT %s",
            (query, TOP_K),
        ).fetchall()
        return [str(row[0]) for row in rows]

    return search, build


def qdrant_native_engine(dataset: BeirDataset) -> tuple[Search, float]:
    client = QdrantClient(url=QDRANT_URL, timeout=120, check_compatibility=False)
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(COLLECTION, vectors_config={})
    start = time.perf_counter()
    client.create_payload_index(
        COLLECTION,
        "body",
        field_schema=models.TextIndexParams(
            type=models.TextIndexType.TEXT,
            tokenizer=models.TokenizerType.WORD,
            lowercase=True,
            stopwords=models.Language.ENGLISH,
            stemmer=models.SnowballParams(type=models.Snowball.SNOWBALL, language=models.SnowballLanguage.ENGLISH),
            scoring=True,
        ),
    )
    ids = sorted(dataset.corpus)
    for begin in range(0, len(ids), 500):
        batch = ids[begin : begin + 500]
        client.upsert(
            COLLECTION,
            points=[
                models.PointStruct(id=int(i), vector={}, payload={"doc_id": d, "body": dataset.corpus[d].full_text})
                for i, d in zip(range(begin, begin + len(batch)), batch, strict=True)
            ],
            wait=True,
        )
    build = time.perf_counter() - start

    def search(query: str) -> list[str]:
        response = client.query_points(
            COLLECTION,
            query=models.TextQuery(text=models.TextQueryInput(query=query)),
            using="body",
            limit=TOP_K,
            with_payload=["doc_id"],
        )
        return [str((p.payload or {})["doc_id"]) for p in response.points]

    return search, build


ENGINES: dict[str, Callable[[BeirDataset], tuple[Search, float]]] = {
    "bm25s": bm25s_engine,
    "paradedb": paradedb_engine,
    "qdrant_native": qdrant_native_engine,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--engines", nargs="+", default=list(ENGINES))
    args = parser.parse_args()
    root = Path(os.environ.get("MEZVUAT_DATA_ROOT", "data")) / "beir"
    directory, _ = download_dataset("scifact", root)
    dataset = load_dataset(directory, "scifact")
    _, qrels = split_dev_from_train(dataset.qrels["train"])

    output: dict[str, object] = {"dataset": "scifact", "split": "dev", "n_queries": len(qrels), "engines": {}}
    per_query: dict[str, dict[str, dict[str, float]]] = {}
    for name in args.engines:
        search, build_seconds = ENGINES[name](dataset)
        latencies: list[float] = []
        run: dict[str, list[str]] = {}
        for query_id in sorted(qrels):
            start = time.perf_counter()
            run[query_id] = search(dataset.queries[query_id])
            latencies.append((time.perf_counter() - start) * 1000)
        metrics: dict[str, dict[str, object]] = {}
        per_query[name] = {}
        for metric in DEFAULT_METRICS:
            values = per_query_metric(qrels, run, metric)
            per_query[name][metric] = values
            metrics[metric] = {
                "mean": float(np.mean(list(values.values()))),
                "ci95": bootstrap_ci(list(values.values())),
            }
        output["engines"][name] = {  # type: ignore[index]
            "metrics": metrics,
            "build_seconds": build_seconds,
            "latency_p50_ms": float(np.percentile(latencies, 50)),
            "latency_p95_ms": float(np.percentile(latencies, 95)),
        }
    comparisons = []
    base = "bm25s"
    for name in args.engines:
        if name != base and base in per_query:
            for metric in ("ndcg@10", "recall@100"):
                outcome = paired_comparison(per_query[base][metric], per_query[name][metric], metric=metric)
                comparisons.append(
                    {
                        "pair": f"{name} - {base}",
                        "metric": metric,
                        "delta": outcome.delta,
                        "ci95": [outcome.ci_low, outcome.ci_high],
                        "permutation_p": outcome.permutation_p,
                    }
                )
    output["comparisons"] = comparisons
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=1), encoding="utf-8")
    print(json.dumps(output, indent=1))  # noqa: T201


if __name__ == "__main__":
    main()
