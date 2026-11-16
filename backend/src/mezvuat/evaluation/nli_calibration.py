"""Calibrate the NLI thresholds τ_e and τ_c on the owner's labels (ADR-0009, M5 part 3).

Run it only once the owner has labelled the pairs (`review nli`): `mezvuat-dataset nli-calibrate`. It scores every
labelled pair with the pinned model on CPU, then searches a grid of thresholds with the verifier's own decision rule
(`NliThresholds.label`). The objective is macro-F1 over supported / contradicted / not enough info, subject to a
minimum precision for `supported`: a false "supported" is the costly error, because it hides a gap in the evidence.
Ties go to the higher (stricter) thresholds. Stratified k-fold cross-validation, fitting on the other folds each
time, estimates how the chosen rule does on pairs it was not tuned on.

The functions here take scores and labels as arguments, so tests can exercise them on synthetic data. The command
reads labels only from the owner's review file, and refuses to run without enough of each label.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mezvuat.citations.support import SUPPORT_LABELS, NliScores, NliThresholds, SupportLabel

# Review labels to verifier labels.
REVIEW_TO_SUPPORT: dict[str, SupportLabel] = {
    "supported": "supported",
    "contradicted": "contradicted",
    "neutral": "not_enough_info",
}
GRID = tuple(round(0.05 * i, 2) for i in range(2, 20))  # 0.10 ... 0.95
MIN_PER_LABEL = 20
DEFAULT_MIN_SUPPORTED_PRECISION = 0.9


class InsufficientLabelsError(ValueError):
    pass


@dataclass(frozen=True)
class Example:
    pair_id: str
    scores: NliScores
    gold: SupportLabel


@dataclass(frozen=True)
class Evaluation:
    n: int
    accuracy: float
    macro_f1: float
    per_label: dict[str, dict[str, float]]
    confusion: dict[str, dict[str, int]]

    def precision(self, label: str) -> float:
        return self.per_label[label]["precision"]

    def to_json(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "per_label": {k: {m: round(v, 4) for m, v in d.items()} for k, d in self.per_label.items()},
            "confusion_gold_by_predicted": self.confusion,
        }


def examples_from(
    labels: dict[str, dict[str, Any]], scores: dict[str, NliScores]
) -> tuple[list[Example], dict[str, int]]:
    """Owner labels joined with model scores. `unusable` pairs and pairs without scores are left out and counted."""
    examples: list[Example] = []
    skipped: Counter[str] = Counter()
    for pair_id, record in sorted(labels.items()):
        gold = REVIEW_TO_SUPPORT.get(record["label"])
        if gold is None:
            skipped[str(record["label"])] += 1
        elif pair_id not in scores:
            skipped["no_score"] += 1
        else:
            examples.append(Example(pair_id, scores[pair_id], gold))
    return examples, dict(skipped)


def check_counts(counts: Mapping[str, int], minimum: int = MIN_PER_LABEL) -> None:
    """Counts by verifier label (`not_enough_info`, not the review's `neutral`)."""
    if any(counts.get(label, 0) < minimum for label in SUPPORT_LABELS):
        have = {label: counts.get(label, 0) for label in SUPPORT_LABELS}
        raise InsufficientLabelsError(f"calibration needs at least {minimum} labelled pairs of each label; have {have}")


def check_enough(examples: Sequence[Example], minimum: int = MIN_PER_LABEL) -> None:
    check_counts(Counter(e.gold for e in examples), minimum)


def evaluate(examples: Sequence[Example], thresholds: NliThresholds) -> Evaluation:
    return score_predictions([(e.gold, thresholds.label(e.scores)) for e in examples])


def score_predictions(pairs: Sequence[tuple[SupportLabel, SupportLabel]]) -> Evaluation:
    """Accuracy, macro-F1 and per-label precision and recall for (gold, predicted) pairs."""
    confusion: dict[str, dict[str, int]] = {g: dict.fromkeys(SUPPORT_LABELS, 0) for g in SUPPORT_LABELS}
    for gold, predicted in pairs:
        confusion[gold][predicted] += 1
    per_label: dict[str, dict[str, float]] = {}
    for label in SUPPORT_LABELS:
        tp = confusion[label][label]
        predicted_n = sum(confusion[g][label] for g in SUPPORT_LABELS)
        actual = sum(confusion[label].values())
        precision = tp / predicted_n if predicted_n else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": float(actual)}
    n = len(pairs)
    correct = sum(confusion[label][label] for label in SUPPORT_LABELS)
    return Evaluation(
        n=n,
        accuracy=correct / n if n else 0.0,
        macro_f1=sum(d["f1"] for d in per_label.values()) / len(per_label),
        per_label=per_label,
        confusion=confusion,
    )


@dataclass(frozen=True)
class Fit:
    thresholds: NliThresholds
    evaluation: Evaluation
    precision_target_met: bool


def fit_thresholds(
    examples: Sequence[Example],
    *,
    min_supported_precision: float | None = DEFAULT_MIN_SUPPORTED_PRECISION,
    grid: Sequence[float] = GRID,
    source: str = "calibrated",
) -> Fit:
    """The grid point with the best macro-F1 among those meeting the precision target (if any meets it)."""
    if not examples:
        raise InsufficientLabelsError("no labelled examples")
    best: tuple[tuple[bool, float, float, float], NliThresholds, Evaluation] | None = None
    for tau_e in grid:
        for tau_c in grid:
            candidate = NliThresholds(entailment=tau_e, contradiction=tau_c, calibrated=True, source=source)
            result = evaluate(examples, candidate)
            meets = min_supported_precision is None or result.precision("supported") >= min_supported_precision
            key = (meets, round(result.macro_f1, 9), tau_e, tau_c)
            if best is None or key > best[0]:
                best = (key, candidate, result)
    if best is None:
        raise InsufficientLabelsError("empty threshold grid")
    return Fit(thresholds=best[1], evaluation=best[2], precision_target_met=best[0][0])


def stratified_folds(examples: Sequence[Example], k: int, seed: int) -> list[list[Example]]:
    rng = random.Random(f"nli-folds:{seed}")  # noqa: S311  (reproducible split)
    folds: list[list[Example]] = [[] for _ in range(k)]
    for label in SUPPORT_LABELS:
        members = sorted((e for e in examples if e.gold == label), key=lambda e: e.pair_id)
        rng.shuffle(members)
        for n, example in enumerate(members):
            folds[n % k].append(example)
    return folds


def cross_validate(
    examples: Sequence[Example],
    *,
    k: int = 5,
    seed: int = 13,
    min_supported_precision: float | None = DEFAULT_MIN_SUPPORTED_PRECISION,
    grid: Sequence[float] = GRID,
) -> dict[str, Any]:
    """Fit on k-1 folds, predict the held-out fold; report the pooled held-out evaluation and the chosen τ per fold."""
    folds = stratified_folds(examples, k, seed)
    held_out: list[tuple[Example, SupportLabel]] = []
    chosen: list[dict[str, float]] = []
    for i, test in enumerate(folds):
        train = [e for j, fold in enumerate(folds) if j != i for e in fold]
        fit = fit_thresholds(train, min_supported_precision=min_supported_precision, grid=grid)
        chosen.append({"entailment": fit.thresholds.entailment, "contradiction": fit.thresholds.contradiction})
        held_out += [(e, fit.thresholds.label(e.scores)) for e in test]
    pooled = score_predictions([(example.gold, predicted) for example, predicted in held_out])
    return {"folds": k, "seed": seed, "held_out": pooled.to_json(), "thresholds_per_fold": chosen}


def calibrate(
    examples: Sequence[Example],
    *,
    min_supported_precision: float | None = DEFAULT_MIN_SUPPORTED_PRECISION,
    folds: int = 5,
    seed: int = 13,
    source: str,
    minimum_per_label: int = MIN_PER_LABEL,
) -> dict[str, Any]:
    """The full procedure: enough labels, the fit on all of them, the provisional rule for comparison, and CV."""
    check_enough(examples, minimum_per_label)
    fit = fit_thresholds(examples, min_supported_precision=min_supported_precision, source=source)
    return {
        "entailment": fit.thresholds.entailment,
        "contradiction": fit.thresholds.contradiction,
        "calibrated": True,
        "source": source,
        "objective": "macro-F1 over supported / contradicted / not_enough_info",
        "min_supported_precision": min_supported_precision,
        "precision_target_met": fit.precision_target_met,
        "labels": dict(sorted(Counter(e.gold for e in examples).items())),
        "fit_on_all_labels": fit.evaluation.to_json(),
        "provisional_rule_on_all_labels": evaluate(examples, NliThresholds()).to_json(),
        "cross_validation": cross_validate(
            examples, k=folds, seed=seed, min_supported_precision=min_supported_precision
        ),
    }
