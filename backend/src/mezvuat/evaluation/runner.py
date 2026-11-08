"""Retrieval experiments over BEIR datasets, run through the production pipeline and logged to MLflow."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import platform
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mezvuat.core.config import get_settings
from mezvuat.evaluation.beir import BeirDataset, download_dataset, load_dataset, split_dev_from_train
from mezvuat.evaluation.metrics import (
    DEFAULT_METRICS,
    bootstrap_ci,
    evaluate,
    paired_comparison,
    per_query_metric,
)
from mezvuat.ml.models import CrossEncoderReranker, SentenceTransformerEmbedder, resolve_device
from mezvuat.ml.registry import get_embedder_spec, get_reranker_spec
from mezvuat.persistence.qdrant import DensePoint, QdrantDenseIndex
from mezvuat.retrieval.fusion import DEFAULT_RRF_K
from mezvuat.retrieval.lexical import Bm25Index
from mezvuat.retrieval.pipeline import RetrievalConfig, RetrievalPipeline
from mezvuat.retrieval.types import ChunkPayload, ChunkRecord, Filters, RetrievalMode, ScoredChunk

log = logging.getLogger("mezvuat.exp")
EXPERIMENT_WORKSPACE = "experiment"
_NO_FILTERS = Filters(exclude_section_kinds=frozenset())


@dataclass(frozen=True)
class ConfigSpec:
    name: str
    mode: RetrievalMode
    embedder: str | None = None
    reranker: str | None = None
    rrf_k: int = DEFAULT_RRF_K
    candidate_depth: int = 100
    rerank_depth: int = 50


@dataclass(frozen=True)
class Comparison:
    baseline: str
    candidate: str


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    mlflow_experiment: str
    dataset: str
    split: str
    configs: tuple[ConfigSpec, ...]
    comparisons: tuple[Comparison, ...] = ()
    seed: int = 13
    k: int = 100
    device: str = "auto"
    extra: dict[str, Any] = field(default_factory=dict)


def load_spec(path: Path) -> ExperimentSpec:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["configs"] = tuple(ConfigSpec(**c) for c in data["configs"])
    data["comparisons"] = tuple(Comparison(**c) for c in data.get("comparisons", []))
    return ExperimentSpec(**data)


def _qrels_for_split(dataset: BeirDataset, split: str, seed: int) -> dict[str, dict[str, int]]:
    if split == "dev":
        _, dev = split_dev_from_train(dataset.qrels["train"], seed=seed)
        return dev
    return dataset.qrels[split]


class _SingleIndexLexical:
    def __init__(self, index: Bm25Index) -> None:
        self._index = index

    def search(self, workspace_id: str, query: str, k: int, filters: Filters) -> list[ScoredChunk]:
        return self._index.search(query, k, filters)


class _CorpusStore:
    """The BEIR corpus as a ChunkStore: a real data source for experiments, not a stand-in."""

    def __init__(self, dataset: BeirDataset) -> None:
        self._records = {
            doc.doc_id: ChunkRecord(
                chunk_id=doc.doc_id,
                workspace_id=EXPERIMENT_WORKSPACE,
                document_id=doc.doc_id,
                document_version_id=doc.doc_id,
                text=doc.full_text,
                text_sha256=doc.doc_id,  # corpus ids are unique documents; no cross-version collapse applies
                embed_text=doc.full_text,
                ordinal=0,
                char_start=0,
                char_end=len(doc.full_text),
                page_start=None,
                page_end=None,
                section_path=(),
                section_kind="body",
            )
            for doc in dataset.corpus.values()
        }

    def get_many(self, workspace_id: str, chunk_ids: list[str]) -> dict[str, ChunkRecord]:
        return {cid: self._records[cid] for cid in chunk_ids if cid in self._records}


def _payload(doc_id: str) -> ChunkPayload:
    return ChunkPayload(
        chunk_id=doc_id,
        document_id=doc_id,
        document_version_id=doc_id,
        is_latest_version=True,
        published_at=None,
        section_kind="body",
    )


def _cache_root() -> Path:
    return Path(os.environ.get("MEZVUAT_DATA_ROOT", "data"))


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _collection(dataset: str, model_id: str) -> str:
    slug = model_id.split("/")[-1].lower().replace("_", "-")
    return f"exp__{dataset}__{slug}"


def build_dense_index(
    dataset: BeirDataset, embedder: SentenceTransformerEmbedder
) -> tuple[QdrantDenseIndex, dict[str, float]]:
    """Embed the corpus and load it into Qdrant, reusing a complete collection and its recorded cost."""
    import torch

    index = QdrantDenseIndex.from_settings(get_settings(), _collection(dataset.name, embedder.spec.model_id))
    index.ensure_collection(embedder.dim)
    cost_file = _cache_root() / "exp-cost" / f"{dataset.name}__{embedder.spec.model_id.replace('/', '_')}.json"
    if index.count(EXPERIMENT_WORKSPACE) == len(dataset.corpus) and cost_file.exists():
        return index, json.loads(cost_file.read_text(encoding="utf-8"))
    doc_ids = sorted(dataset.corpus)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    vectors = embedder.embed_documents([dataset.corpus[d].full_text for d in doc_ids])
    embed_seconds = time.perf_counter() - start
    index.upsert(
        [
            DensePoint(EXPERIMENT_WORKSPACE, _payload(doc_id), [float(x) for x in vector])
            for doc_id, vector in zip(doc_ids, vectors, strict=True)
        ]
    )
    cost = {
        "indexing_seconds": embed_seconds,
        "chunks_per_second": len(doc_ids) / embed_seconds,
        "n_chunks": float(len(doc_ids)),
        "peak_vram_mb": (torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else 0.0,
    }
    cost_file.parent.mkdir(parents=True, exist_ok=True)
    cost_file.write_text(json.dumps(cost), encoding="utf-8")
    return index, cost


def write_run_file(path: Path, run: dict[str, list[tuple[str, float]]], tag: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for query_id in sorted(run):
            for rank, (doc_id, score) in enumerate(run[query_id], start=1):
                handle.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {tag}\n")


def write_per_query(path: Path, table: dict[str, dict[str, float]], metrics: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", *metrics])
        for query_id in sorted(table):
            writer.writerow([query_id, *(f"{table[query_id][m]:.6f}" for m in metrics)])


def _percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def run_experiment(spec: ExperimentSpec, *, artifact_root: Path) -> list[dict[str, Any]]:
    import mlflow

    device = resolve_device(spec.device)
    data_root = _cache_root() / "beir"
    directory, zip_sha = download_dataset(spec.dataset, data_root)
    dataset = load_dataset(directory, spec.dataset)
    qrels = _qrels_for_split(dataset, spec.split, spec.seed)
    log.info("dataset ready", extra={"dataset": spec.dataset, "split": spec.split, "n_queries": len(qrels)})

    lexical_index = Bm25Index.build(
        [_payload(doc_id) for doc_id in sorted(dataset.corpus)],
        [dataset.corpus[doc_id].full_text for doc_id in sorted(dataset.corpus)],
    )
    store = _CorpusStore(dataset)
    embedders: dict[str, SentenceTransformerEmbedder] = {}
    dense_indexes: dict[str, tuple[QdrantDenseIndex, dict[str, float]]] = {}
    rerankers: dict[str, CrossEncoderReranker] = {}

    mlflow.set_experiment(spec.mlflow_experiment)
    git_sha = _git_sha()
    summaries: list[dict[str, Any]] = []
    per_query_by_config: dict[str, dict[str, dict[str, float]]] = {}
    for config in spec.configs:
        embedder_id = config.embedder
        needs_dense = config.mode != "bm25"
        if needs_dense and embedder_id is None:
            raise ValueError(f"config {config.name} needs an embedder")
        query_embedder: SentenceTransformerEmbedder | None = None
        dense: QdrantDenseIndex | None = None
        cost: dict[str, float] = {}
        if needs_dense and embedder_id is not None:
            if embedder_id not in embedders:
                # Free the previous embedder before loading another: only 4 GB of VRAM.
                embedders.clear()
                _release_gpu()
                embedders[embedder_id] = SentenceTransformerEmbedder.from_model_id(
                    embedder_id, device=device, batch_size=spec.extra.get("embed_batch_size", 64)
                )
            query_embedder = embedders[embedder_id]
            if embedder_id not in dense_indexes:
                dense_indexes[embedder_id] = build_dense_index(dataset, query_embedder)
            dense, cost = dense_indexes[embedder_id]
        reranker: CrossEncoderReranker | None = None
        if config.mode == "hybrid_rrf_rerank":
            if config.reranker is None:
                raise ValueError(f"config {config.name} needs a reranker")
            if config.reranker not in rerankers:
                rerankers.clear()
                _release_gpu()
                rerankers[config.reranker] = CrossEncoderReranker.from_model_id(
                    config.reranker, device=device, batch_size=spec.extra.get("rerank_batch_size", 32)
                )
            reranker = rerankers[config.reranker]

        pipeline = RetrievalPipeline(
            lexical=_SingleIndexLexical(lexical_index),
            dense=dense if dense is not None else _NoDense(),
            embedder=query_embedder if query_embedder is not None else _NoEmbedder(),
            store=store,
            reranker=reranker,
        )
        retrieval_config = RetrievalConfig(
            mode=config.mode,
            candidate_depth=max(config.candidate_depth, spec.k),
            rerank_depth=config.rerank_depth,
            rrf_k=config.rrf_k,
            collapse_duplicates=False,
        )
        run: dict[str, list[tuple[str, float]]] = {}
        timings: dict[str, list[float]] = {}
        for query_id in sorted(qrels):
            result = pipeline.retrieve(
                EXPERIMENT_WORKSPACE,
                dataset.queries[query_id],
                k=spec.k,
                config=retrieval_config,
                filters=_NO_FILTERS,
            )
            run[query_id] = [(e.chunk.chunk_id, e.score) for e in result.evidence]
            for stage, milliseconds in result.timings_ms.items():
                timings.setdefault(stage, []).append(milliseconds)

        ranked = {query_id: [doc for doc, _ in rows] for query_id, rows in run.items()}
        metrics = evaluate(qrels, ranked, DEFAULT_METRICS)
        table: dict[str, dict[str, float]] = {query_id: {} for query_id in qrels}
        confidence: dict[str, tuple[float, float]] = {}
        for metric in DEFAULT_METRICS:
            values = per_query_metric(qrels, ranked, metric)
            for query_id, value in values.items():
                table[query_id][metric] = value
            confidence[metric] = bootstrap_ci(list(values.values()), seed=spec.seed)
        per_query_by_config[config.name] = table

        with mlflow.start_run(run_name=f"{spec.name}/{config.name}") as active:
            mlflow.log_params(
                {
                    **{f"config.{k}": str(v) for k, v in asdict(config).items()},
                    "dataset": spec.dataset,
                    "dataset_zip_sha256": zip_sha,
                    "split": spec.split,
                    "n_queries": len(qrels),
                    "n_docs": len(dataset.corpus),
                    "seed": spec.seed,
                    "device": device,
                    "git_sha": git_sha,
                    "machine": platform.processor() or platform.machine(),
                    "embedder_revision": str(get_embedder_spec(embedder_id).revision) if embedder_id else "",
                    "reranker_revision": str(get_reranker_spec(config.reranker).revision) if config.reranker else "",
                    "lock_sha256": _lock_digest(),
                }
            )
            safe_metrics = {m.replace("@", "_at_"): value for m, value in metrics.items()}
            mlflow.log_metrics(safe_metrics)
            mlflow.log_metrics(
                {f"{m.replace('@', '_at_')}_ci_low": low for m, (low, _) in confidence.items()}
                | {f"{m.replace('@', '_at_')}_ci_high": high for m, (_, high) in confidence.items()}
            )
            latency = {f"latency_{stage}_p50_ms": _percentile(values, 50) for stage, values in timings.items()} | {
                f"latency_{stage}_p95_ms": _percentile(values, 95) for stage, values in timings.items()
            }
            mlflow.log_metrics(latency | {f"cost_{name}": value for name, value in cost.items()})
            directory_out = artifact_root / active.info.run_id
            directory_out.mkdir(parents=True, exist_ok=True)
            write_run_file(directory_out / "run.trec", run, config.name)
            write_per_query(directory_out / "per_query.csv", table, DEFAULT_METRICS)
            (directory_out / "config.json").write_text(json.dumps(asdict(config), indent=1), encoding="utf-8")
            mlflow.log_artifacts(str(directory_out))
            summaries.append(
                {
                    "run_id": active.info.run_id,
                    "config": config.name,
                    "dataset": spec.dataset,
                    "split": spec.split,
                    "n_queries": len(qrels),
                    "metrics": metrics,
                    "ci95": confidence,
                    "latency": latency,
                    "cost": cost,
                }
            )
            log.info("config finished", extra={"config": config.name, "metrics": metrics})

    for comparison in spec.comparisons:
        base = per_query_by_config[comparison.baseline]
        candidate = per_query_by_config[comparison.candidate]
        for metric in ("ndcg@10", "recall@100", "mrr@10"):
            outcome = paired_comparison(
                {q: base[q][metric] for q in base},
                {q: candidate[q][metric] for q in candidate},
                metric=metric,
                seed=spec.seed,
            )
            summaries.append({"comparison": f"{comparison.candidate} - {comparison.baseline}", **asdict(outcome)})
            log.info(
                "comparison",
                extra=asdict(outcome) | {"pair": f"{comparison.candidate} - {comparison.baseline}"},
            )
    embedders.clear()
    rerankers.clear()
    return summaries


class _NoDense:
    def search(self, workspace_id: str, query_vector: list[float], k: int, filters: Filters) -> list[ScoredChunk]:
        raise AssertionError("dense retriever used in a lexical-only configuration")


class _NoEmbedder:
    def embed_query(self, query: str) -> list[float]:
        raise AssertionError("embedder used in a lexical-only configuration")


def _release_gpu() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _lock_digest() -> str:
    lock = Path("uv.lock")
    return hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else "unknown"
