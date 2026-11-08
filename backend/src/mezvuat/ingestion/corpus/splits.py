"""Split the demo corpus by document family before any query is generated (ADR-0007).

A family is every version of an arXiv id. Families whose text substantially overlaps are merged so that the same
content can never appear on both sides of a split.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_WORD = re.compile(r"[a-z0-9]+")
JACCARD_MERGE_THRESHOLD = 0.8
SPLIT_FRACTIONS = (("train", 0.6), ("val", 0.2), ("test", 0.2))


@dataclass(frozen=True)
class Family:
    arxiv_id: str
    stratum: str
    text: str


def shingles(text: str, n: int = 5) -> set[int]:
    tokens = _WORD.findall(text.lower())
    return {hash(tuple(tokens[i : i + n])) for i in range(len(tokens) - n + 1)}


def jaccard(left: set[int], right: set[int]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def merge_overlapping(families: Sequence[Family], threshold: float = JACCARD_MERGE_THRESHOLD) -> list[list[Family]]:
    """Group families connected by text overlap at or above `threshold` (union-find)."""
    parent = list(range(len(families)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    sets = [shingles(family.text) for family in families]
    for i in range(len(families)):
        for j in range(i + 1, len(families)):
            if jaccard(sets[i], sets[j]) >= threshold:
                parent[find(j)] = find(i)
    groups: dict[int, list[Family]] = defaultdict(list)
    for index, family in enumerate(families):
        groups[find(index)].append(family)
    return sorted(groups.values(), key=lambda group: min(f.arxiv_id for f in group))


def assign_splits(groups: Sequence[Sequence[Family]], *, seed: int = 13) -> dict[str, str]:
    """Assign whole groups to train/val/test, stratified by each group's first family stratum."""
    by_stratum: dict[str, list[Sequence[Family]]] = defaultdict(list)
    for group in groups:
        by_stratum[min(f.stratum for f in group)].append(group)
    rng = random.Random(seed)  # noqa: S311  (reproducible split, not security)
    assignment: dict[str, str] = {}
    counts = dict.fromkeys((name for name, _ in SPLIT_FRACTIONS), 0)
    total = sum(len(group) for group in groups)
    for stratum in sorted(by_stratum):
        members = sorted(by_stratum[stratum], key=lambda g: min(f.arxiv_id for f in g))
        rng.shuffle(members)
        for group in members:
            # Largest remaining deficit against the target fraction wins, so small strata stay balanced.
            name = max(SPLIT_FRACTIONS, key=lambda item: item[1] * total - counts[item[0]])[0]
            for family in group:
                assignment[family.arxiv_id] = name
            counts[name] += len(group)
    return assignment


def build_splits(families: Sequence[Family], *, seed: int = 13) -> Mapping[str, object]:
    groups = merge_overlapping(families)
    assignment = assign_splits(groups, seed=seed)
    return {
        "schema": 1,
        "seed": seed,
        "jaccard_merge_threshold": JACCARD_MERGE_THRESHOLD,
        "merged_groups": [sorted(f.arxiv_id for f in group) for group in groups if len(group) > 1],
        "assignment": dict(sorted(assignment.items())),
    }
