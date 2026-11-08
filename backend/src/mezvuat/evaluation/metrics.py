"""Retrieval metrics and paired bootstrap comparison (ADR-0007). Cross-checked against ranx in the tests."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

Qrels = Mapping[str, Mapping[str, int]]
Run = Mapping[str, Sequence[str]]


def recall_at_k(relevant: Mapping[str, int], ranked: Sequence[str], k: int) -> float:
    positives = {doc for doc, grade in relevant.items() if grade > 0}
    if not positives:
        return 0.0
    return len(positives.intersection(ranked[:k])) / len(positives)


def precision_at_k(relevant: Mapping[str, int], ranked: Sequence[str], k: int) -> float:
    return sum(1 for doc in ranked[:k] if relevant.get(doc, 0) > 0) / k


def mrr_at_k(relevant: Mapping[str, int], ranked: Sequence[str], k: int) -> float:
    for rank, doc in enumerate(ranked[:k], start=1):
        if relevant.get(doc, 0) > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant: Mapping[str, int], ranked: Sequence[str], k: int) -> float:
    """Linear gains, the BEIR/pytrec_eval convention: gain = grade / log2(rank + 1)."""
    dcg = sum(relevant.get(doc, 0) / math.log2(rank + 1) for rank, doc in enumerate(ranked[:k], start=1))
    ideal = sorted((g for g in relevant.values() if g > 0), reverse=True)[:k]
    idcg = sum(grade / math.log2(rank + 1) for rank, grade in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


DEFAULT_METRICS: tuple[str, ...] = (
    "recall@10",
    "recall@50",
    "recall@100",
    "precision@5",
    "precision@10",
    "mrr@10",
    "ndcg@10",
)


def per_query_metric(qrels: Qrels, run: Run, metric: str) -> dict[str, float]:
    name, _, cutoff = metric.partition("@")
    k = int(cutoff)
    functions = {"recall": recall_at_k, "precision": precision_at_k, "mrr": mrr_at_k, "ndcg": ndcg_at_k}
    fn = functions[name]
    return {qid: fn(qrels[qid], list(run.get(qid, ())), k) for qid in qrels}


def evaluate(qrels: Qrels, run: Run, metrics: Sequence[str] = DEFAULT_METRICS) -> dict[str, float]:
    return {m: float(np.mean(list(per_query_metric(qrels, run, m).values()))) for m in metrics}


@dataclass(frozen=True)
class PairedComparison:
    metric: str
    n_queries: int
    mean_a: float
    mean_b: float
    delta: float
    ci_low: float
    ci_high: float
    permutation_p: float


def bootstrap_ci(values: Sequence[float], *, resamples: int = 10_000, seed: int = 13) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean, over queries."""
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(resamples, len(array)))
    means = array[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def paired_comparison(
    per_query_a: Mapping[str, float],
    per_query_b: Mapping[str, float],
    *,
    metric: str = "ndcg@10",
    resamples: int = 10_000,
    seed: int = 13,
) -> PairedComparison:
    """Paired bootstrap CI of mean(B - A) and a two-sided sign-flip randomisation test."""
    if per_query_a.keys() != per_query_b.keys():
        raise ValueError("comparison requires the same queries on both sides")
    queries = sorted(per_query_a)
    a = np.array([per_query_a[q] for q in queries], dtype=np.float64)
    b = np.array([per_query_b[q] for q in queries], dtype=np.float64)
    diffs = b - a
    low, high = bootstrap_ci(diffs.tolist(), resamples=resamples, seed=seed)
    rng = np.random.default_rng(seed + 1)
    signs = rng.choice([-1.0, 1.0], size=(resamples, len(diffs)))
    observed = abs(float(diffs.mean()))
    extreme = int((np.abs((signs * diffs).mean(axis=1)) >= observed - 1e-12).sum())
    p_value = (extreme + 1) / (resamples + 1)
    return PairedComparison(
        metric=metric,
        n_queries=len(queries),
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        delta=float(diffs.mean()),
        ci_low=low,
        ci_high=high,
        permutation_p=p_value,
    )
