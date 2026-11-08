from __future__ import annotations

from collections import Counter

from mezvuat.ingestion.corpus.splits import Family, assign_splits, build_splits, jaccard, merge_overlapping, shingles


def family(arxiv_id: str, text: str | None = None, stratum: str = "cs.IR/2024") -> Family:
    return Family(arxiv_id, stratum, text or f"unique words for document {arxiv_id} " * 20)


def test_jaccard_of_identical_and_disjoint_text() -> None:
    left = shingles("alpha beta gamma delta epsilon zeta eta")
    assert jaccard(left, left) == 1.0
    assert jaccard(left, shingles("one two three four five six seven")) == 0.0
    assert jaccard(set(), left) == 0.0


def test_overlapping_families_are_merged_transitively() -> None:
    shared = " ".join(f"token{i}" for i in range(500))
    families = [
        family("a", shared),
        family("b", shared + " tail"),
        family("c", " ".join(f"other{i}" for i in range(500))),
    ]
    groups = merge_overlapping(families)
    assert sorted(sorted(f.arxiv_id for f in g) for g in groups) == [["a", "b"], ["c"]]


def test_merged_families_never_cross_splits() -> None:
    shared = " ".join(f"token{i}" for i in range(500))
    families = [family(f"id{i}", shared + f" tail{i}") for i in range(4)]
    families += [family(f"solo{i}") for i in range(40)]
    assignment = build_splits(families)["assignment"]
    assert isinstance(assignment, dict)
    assert len({assignment[f"id{i}"] for i in range(4)}) == 1


def test_split_proportions_are_close_to_60_20_20() -> None:
    groups = [[family(f"id{i:03d}", stratum=f"s{i % 4}")] for i in range(200)]
    counts = Counter(assign_splits(groups).values())
    assert counts == {"train": 120, "val": 40, "test": 40}


def test_split_is_deterministic_given_a_seed() -> None:
    groups = [[family(f"id{i:03d}")] for i in range(50)]
    assert assign_splits(groups, seed=7) == assign_splits(list(reversed(groups)), seed=7)
    assert assign_splits(groups, seed=7) != assign_splits(groups, seed=8)
