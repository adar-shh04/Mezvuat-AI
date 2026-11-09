from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from mezvuat.ingestion.corpus.manifest import ALLOWED_LICENSES, is_eligible, load_allowlist, select_papers
from mezvuat.ingestion.corpus.oai import ArxivRecord, ArxivVersion, RateLimiter, parse_abs_page, parse_search_entries

CC_BY = "http://creativecommons.org/licenses/by/4.0/"
NON_EXCLUSIVE = "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"

ABS_PAGE = f"""
<div class="submission-history"><h2>Submission history</h2>From: A Author
<strong><a href="/abs/2401.00001v1">[v1]</a></strong> Mon, 1 Jan 2024 09:00:00 UTC (90 KB)<br/>
<strong>[v2]</strong> Tue, 6 Feb 2024 10:00:00 UTC (100 KB)<br/></div>
<div class="abs-license"><a href="{CC_BY}" title="Rights to this article" class="has_license">view</a></div>
"""

FEED = b"""<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
<id>https://arxiv.org/api/x</id>
<entry><id>http://arxiv.org/abs/2401.00001v2</id><title>Hybrid  retrieval
 for RAG</title><summary> We study rerank models. </summary>
<author><name>A. Author</name></author><author><name>B. Author</name></author>
<arxiv:primary_category term="cs.IR"/><category term="cs.IR"/><category term="cs.CL"/></entry>
</feed>"""


def test_search_feed_is_parsed_without_version_suffix() -> None:
    entry = parse_search_entries(FEED)[0]
    assert entry.arxiv_id == "2401.00001"
    assert entry.title == "Hybrid retrieval for RAG"
    assert entry.authors == "A. Author, B. Author"
    assert entry.categories == ("cs.IR", "cs.CL")


def test_abs_page_gives_licence_and_ordered_versions() -> None:
    license_url, versions = parse_abs_page(ABS_PAGE)
    assert license_url == CC_BY
    assert [(v.label, v.submitted) for v in versions] == [
        ("v1", date(2024, 1, 1)),
        ("v2", date(2024, 2, 6)),
    ]


def test_abs_page_without_a_licence_link_means_not_cc() -> None:
    license_url, _ = parse_abs_page("<strong>[v1]</strong> Mon, 1 Jan 2024 09:00:00 UTC (1 KB)")
    assert license_url is None


def test_xml_entity_expansion_is_refused() -> None:
    from defusedxml import EntitiesForbidden

    with pytest.raises(EntitiesForbidden):
        parse_search_entries(b'<!DOCTYPE x [<!ENTITY a "aaaa">]><x>&a;</x>')


def test_rate_limiter_keeps_three_seconds_between_requests() -> None:
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(3.0, monotonic=lambda: now[0], sleep=sleep)
    limiter.wait()
    now[0] += 1.0
    limiter.wait()
    limiter.wait()
    assert sleeps == [pytest.approx(2.0), pytest.approx(3.0)]


def record(arxiv_id: str, *, license_url: str = CC_BY, versions: int = 2, text: str = "rerank") -> ArxivRecord:
    return ArxivRecord(
        arxiv_id=arxiv_id,
        title=text,
        authors="A",
        categories=("cs.IR",),
        license=license_url,
        abstract="",
        versions=tuple(ArxivVersion(f"v{i}", i, date(2024, 1, i)) for i in range(1, versions + 1)),
    )


def test_eligibility_requires_licence_category_and_topic() -> None:
    assert is_eligible(record("1"))
    assert not is_eligible(record("1", license_url=NON_EXCLUSIVE))
    assert not is_eligible(record("1", text="protein folding"))
    assert NON_EXCLUSIVE not in ALLOWED_LICENSES


def test_selection_is_deterministic_and_keeps_multi_version_share() -> None:
    records = [record(f"m{i}") for i in range(30)] + [record(f"s{i}", versions=1) for i in range(30)]
    chosen = select_papers(records, target=20, min_multi_version_share=0.4)
    assert chosen == select_papers(list(reversed(records)), target=20, min_multi_version_share=0.4)
    assert len(chosen) == 20
    assert sum(1 for r in chosen if len(r.versions) >= 2) >= 8


def test_allowlist_only_contains_downloaded_cc_versions(tmp_path: Path) -> None:
    manifest = {
        "papers": [
            {
                "arxiv_id": "2401.00001",
                "title": "T",
                "authors": "A",
                "license": CC_BY,
                "versions": [
                    {"label": "v1", "ordinal": 1, "submitted": "2024-01-01", "raw_sha256": "aa"},
                    {"label": "v2", "ordinal": 2, "submitted": "2024-02-01", "raw_sha256": "bb"},
                    {"label": "v3", "ordinal": 3, "submitted": "2024-03-01"},
                ],
            },
            {
                "arxiv_id": "2401.00002",
                "title": "T",
                "authors": "A",
                "license": NON_EXCLUSIVE,
                "versions": [{"label": "v1", "ordinal": 1, "submitted": "2024-01-01", "raw_sha256": "cc"}],
            },
        ]
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    allowlist = load_allowlist(path)
    assert set(allowlist) == {"aa", "bb"}
    assert allowlist["bb"].first_submitted == date(2024, 1, 1)
    assert allowlist["bb"].latest_ordinal == 3


def test_missing_manifest_means_empty_allowlist(tmp_path: Path) -> None:
    assert load_allowlist(tmp_path / "absent.json") == {}
