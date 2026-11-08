from __future__ import annotations

import pytest
from ranx import Qrels, Run
from ranx import evaluate as ranx_evaluate

from mezvuat.evaluation.metrics import (
    bootstrap_ci,
    evaluate,
    mrr_at_k,
    ndcg_at_k,
    paired_comparison,
    recall_at_k,
)

QRELS = {
    "q1": {"d1": 2, "d2": 1, "d7": 1},
    "q2": {"d3": 1},
    "q3": {"d4": 2, "d5": 2},
}
RUN = {
    "q1": ["d9", "d1", "d2", "d8", "d7"],
    "q2": ["d6", "d8", "d9", "d1", "d2"],
    "q3": ["d5", "d4", "d1", "d2", "d3"],
}


@pytest.mark.parametrize(
    ("ours", "theirs"),
    [
        ("ndcg@3", "ndcg@3"),
        ("recall@3", "recall@3"),
        ("precision@3", "precision@3"),
        ("mrr@5", "mrr@5"),
        ("recall@5", "recall@5"),
    ],
)
def test_metrics_agree_with_ranx(ours: str, theirs: str) -> None:
    ranx_qrels = Qrels({q: dict(d) for q, d in QRELS.items()})
    ranx_run = Run({q: {doc: float(len(docs) - i) for i, doc in enumerate(docs)} for q, docs in RUN.items()})
    assert evaluate(QRELS, RUN, [ours])[ours] == pytest.approx(float(ranx_evaluate(ranx_qrels, ranx_run, theirs)))


def test_zero_relevant_retrieved() -> None:
    assert ndcg_at_k({"a": 1}, ["b"], 10) == 0.0
    assert mrr_at_k({"a": 1}, ["b"], 10) == 0.0
    assert recall_at_k({"a": 1}, ["b"], 10) == 0.0


def test_bootstrap_ci_contains_the_mean() -> None:
    low, high = bootstrap_ci([0.0, 1.0] * 50)
    assert low < 0.5 < high


def test_bootstrap_ci_coverage_on_simulated_data() -> None:
    import numpy as np

    rng = np.random.default_rng(1)
    trials, covered = 200, 0
    true_mean = 0.3
    for trial in range(trials):
        sample = rng.normal(true_mean, 0.2, size=100)
        low, high = bootstrap_ci(sample.tolist(), resamples=800, seed=trial)
        covered += int(low <= true_mean <= high)
    assert 0.90 <= covered / trials <= 0.99


def test_paired_comparison_detects_a_real_difference() -> None:
    import numpy as np

    rng = np.random.default_rng(3)
    a = {str(i): float(v) for i, v in enumerate(rng.uniform(0, 0.5, 200))}
    b = {q: v + 0.1 for q, v in a.items()}
    outcome = paired_comparison(a, b)
    assert outcome.delta == pytest.approx(0.1)
    assert outcome.ci_low > 0
    assert outcome.permutation_p < 0.01


def test_paired_comparison_finds_nothing_in_noise() -> None:
    import numpy as np

    rng = np.random.default_rng(5)
    a = {str(i): float(v) for i, v in enumerate(rng.uniform(0, 1, 300))}
    b = {q: float(v + rng.normal(0, 0.05)) for q, v in a.items()}
    outcome = paired_comparison(a, b)
    assert outcome.ci_low < 0 < outcome.ci_high


def test_paired_comparison_requires_same_queries() -> None:
    with pytest.raises(ValueError):
        paired_comparison({"a": 1.0}, {"b": 1.0})
