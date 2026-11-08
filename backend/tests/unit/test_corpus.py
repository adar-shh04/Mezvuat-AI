from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from mezvuat.ingestion.corpus.manifest import ALLOWED_LICENSES, is_eligible, load_allowlist, select_papers
from mezvuat.ingestion.corpus.oai import ArxivRecord, ArxivVersion, RateLimiter, parse_list_records

CC_BY = "http://creativecommons.org/licenses/by/4.0/"
NON_EXCLUSIVE = "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"

PAGE = f"""<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
<ListRecords>
<record><header><identifier>oai:arXiv.org:2401.00001</identifier></header>
<metadata><arXivRaw xmlns="http://arxiv.org/OAI/arXivRaw/">
<id>2401.00001</id>
<version version="v2"><date>Tue, 6 Feb 2024 10:00:00 GMT</date><size>100kb</size></version>
<version version="v1"><date>Mon, 1 Jan 2024 09:00:00 GMT</date><size>90kb</size></version>
<title>Hybrid  retrieval
 for RAG</title><authors>A. Author, B. Author</authors><categories>cs.IR cs.CL</categories>
<license>{CC_BY}</license><abstract>  We study rerank models.  </abstract>
</arXivRaw></metadata></record>
<record><header status="deleted"><identifier>oai:arXiv.org:gone</identifier></header></record>
<resumptionToken>tok123</resumptionToken>
</ListRecords></OAI-PMH>
""".encode()


def test_parse_page_extracts_versions_in_order_and_resumption_token() -> None:
    records, token = parse_list_records(PAGE)
    assert token == "tok123"
    assert len(records) == 1
    record = records[0]
    assert record.arxiv_id == "2401.00001"
    assert record.title == "Hybrid retrieval for RAG"
    assert record.license == CC_BY
    assert record.categories == ("cs.IR", "cs.CL")
    assert [(v.label, v.submitted) for v in record.versions] == [
        ("v1", date(2024, 1, 1)),
        ("v2", date(2024, 2, 6)),
    ]


def test_oai_error_raises_but_no_records_is_empty() -> None:
    empty = b'<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><error code="noRecordsMatch"/></OAI-PMH>'
    assert parse_list_records(empty) == ([], None)
    bad = b'<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><error code="badArgument">no</error></OAI-PMH>'
    with pytest.raises(RuntimeError):
        parse_list_records(bad)


def test_xml_entity_expansion_is_refused() -> None:
    from defusedxml import EntitiesForbidden

    bomb = b'<!DOCTYPE x [<!ENTITY a "aaaa">]><x>&a;</x>'
    with pytest.raises(EntitiesForbidden):
        parse_list_records(bomb)


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
