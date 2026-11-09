"""Demo corpus manifest: selection, and the allowlist that public-workspace verification relies on (decision D1).

The manifest is committed; the PDFs are not. A document counts as public only when the SHA-256 of its bytes is in
this manifest, so public status can never be claimed by a caller-supplied flag.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from mezvuat.ingestion.corpus.oai import ArxivRecord

ALLOWED_LICENSES = frozenset(
    {
        "http://creativecommons.org/licenses/by/4.0/",
        "http://creativecommons.org/licenses/by-sa/4.0/",
        "http://creativecommons.org/publicdomain/zero/1.0/",
    }
)
TARGET_CATEGORIES = frozenset({"cs.IR", "cs.CL"})
TOPIC_KEYWORDS = (
    "retrieval-augmented",
    "retrieval augmented",
    "rag ",
    "dense retrieval",
    "rerank",
    "re-rank",
    "hybrid retrieval",
    "passage retrieval",
    "information retrieval",
    "cross-encoder",
    "bm25",
    "embedding model",
)


@dataclass(frozen=True)
class ManifestVersion:
    arxiv_id: str
    label: str
    ordinal: int
    submitted: date
    raw_sha256: str
    license: str
    title: str
    authors: str
    latest_ordinal: int
    first_submitted: date


def topic_score(record: ArxivRecord) -> int:
    haystack = f"{record.title} {record.abstract}".lower()
    return sum(haystack.count(keyword) for keyword in TOPIC_KEYWORDS)


def is_eligible(record: ArxivRecord) -> bool:
    return (
        record.license in ALLOWED_LICENSES
        and bool(TARGET_CATEGORIES.intersection(record.categories))
        and topic_score(record) > 0
    )


def select_papers(
    records: Iterable[ArxivRecord], *, target: int = 150, min_multi_version_share: float = 0.4
) -> list[ArxivRecord]:
    """Deterministically pick eligible papers, preferring topic relevance and multi-version papers.

    At least `min_multi_version_share` of the selection has two or more versions, when enough exist.
    """
    unique = {record.arxiv_id: record for record in records if is_eligible(record)}
    ranked = sorted(unique.values(), key=lambda r: (-topic_score(r), r.arxiv_id))
    multi = [r for r in ranked if len(r.versions) >= 2]
    single = [r for r in ranked if len(r.versions) < 2]
    multi_quota = max(int(target * min_multi_version_share), min(len(multi), target - min(len(single), target // 2)))
    chosen = multi[:multi_quota]
    chosen += single[: target - len(chosen)]
    return sorted(chosen, key=lambda r: r.arxiv_id)


def manifest_from_records(records: Iterable[ArxivRecord]) -> dict[str, Any]:
    return {"schema": 1, "source": "arXiv API and abstract pages", "papers": [r.to_json() for r in records]}


def load_allowlist(path: Path) -> dict[str, ManifestVersion]:
    """Map raw_sha256 -> version for every downloaded version in the manifest. Missing file: empty allowlist."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    allowlist: dict[str, ManifestVersion] = {}
    for paper in data["papers"]:
        if paper["license"] not in ALLOWED_LICENSES:
            continue
        latest = max(v["ordinal"] for v in paper["versions"])
        first = min(paper["versions"], key=lambda v: v["ordinal"])
        for version in paper["versions"]:
            digest = version.get("raw_sha256")
            if digest:
                allowlist[digest] = ManifestVersion(
                    arxiv_id=paper["arxiv_id"],
                    label=version["label"],
                    ordinal=version["ordinal"],
                    submitted=date.fromisoformat(version["submitted"]),
                    raw_sha256=digest,
                    license=paper["license"],
                    title=paper["title"],
                    authors=paper["authors"],
                    latest_ordinal=latest,
                    first_submitted=date.fromisoformat(first["submitted"]),
                )
    return allowlist


@lru_cache(maxsize=4)
def cached_allowlist(path: Path) -> dict[str, ManifestVersion]:
    return load_allowlist(path)
