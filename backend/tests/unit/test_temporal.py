"""Temporal reasoning (ADR-0010, M5 acceptance D9): as-of resolution, latest version, dates by kind in answers."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date
from typing import Any

import pydantic
import pytest

from mezvuat.inference import LLMRequest
from mezvuat.orchestration import Budget, ResearchState, RunFilters, WorkflowDeps, build_graph, run_to_end
from mezvuat.temporal import DateFact, VersionDates, compare_to, describe_dates, resolve_as_of
from tests.fakes import KeywordSearch, ScriptedLLM, chunk, shown_evidence


def fact(kind: str, value: str, precision: str = "day", source: str = "source_metadata") -> DateFact:
    return DateFact(kind=kind, value=date.fromisoformat(value), precision=precision, source=source, confidence="high")


def version(version_id: str, label: str, *facts: DateFact, latest: bool | None = None) -> VersionDates:
    return VersionDates(
        version_id=version_id,
        version_label=label,
        version_ordinal=int(label[1:]),
        is_latest_version=latest,
        facts=list(facts),
    )


# ----- resolution rules ----------------------------------------------------------------------------------------------


def test_as_of_picks_the_newest_version_dated_on_or_before_the_date() -> None:
    versions = [
        version("v1", "v1", fact("published", "2022-01-10")),
        version("v2", "v2", fact("revised", "2023-05-02")),
        version("v3", "v3", fact("revised", "2024-08-01")),
    ]
    resolved = resolve_as_of(versions, date(2024, 1, 1))
    assert resolved.applicable == "v2"
    assert resolved.not_yet_available == ["v3"]
    assert resolved.unresolved == []


def test_same_year_year_precision_does_not_resolve_an_as_of_query() -> None:
    versions = [
        version("v1", "v1", fact("published", "2022-03-01")),
        version("v2", "v2", fact("revised", "2024-01-01", "year")),
    ]
    resolved = resolve_as_of(versions, date(2024, 6, 1))
    assert resolved.unresolved == ["v2"]
    assert resolved.applicable == "v1"
    # A different year decides it either way.
    assert compare_to(fact("revised", "2023-01-01", "year"), date(2024, 6, 1)) == "before_or_on"
    assert compare_to(fact("revised", "2025-01-01", "year"), date(2024, 6, 1)) == "after"


def test_month_precision_is_undecidable_within_its_month() -> None:
    assert compare_to(fact("published", "2024-06-01", "month"), date(2024, 6, 20)) == "undecidable"
    assert compare_to(fact("published", "2024-05-01", "month"), date(2024, 6, 1)) == "before_or_on"


def test_unknown_dates_are_unresolved_never_guessed() -> None:
    resolved = resolve_as_of([version("v1", "v1"), version("v2", "v2")], date(2024, 1, 1))
    assert resolved.applicable is None and resolved.unresolved == ["v1", "v2"]


def test_two_versions_on_the_same_day_cannot_be_ordered() -> None:
    versions = [
        version("v1", "v1", fact("published", "2024-01-01")),
        version("v2", "v2", fact("revised", "2024-01-01")),
    ]
    resolved = resolve_as_of(versions, date(2024, 2, 1))
    assert resolved.applicable is None and sorted(resolved.unresolved) == ["v1", "v2"]


def test_effective_dates_take_precedence_for_applicability() -> None:
    v1_facts = [
        fact("published", "2020-01-01"),
        fact("effective_from", "2020-01-01"),
        fact("effective_to", "2023-01-01"),
    ]
    versions = [
        version("v1", "v1", *v1_facts),
        version("v2", "v2", fact("published", "2022-06-01"), fact("effective_from", "2023-01-01")),
    ]
    # v2 was published earlier than the date, but it only took effect later.
    assert resolve_as_of(versions, date(2022, 12, 1)).applicable == "v1"
    assert resolve_as_of(versions, date(2023, 6, 1)).applicable == "v2"


def test_ingestion_time_is_not_a_date_kind() -> None:
    with pytest.raises(pydantic.ValidationError):
        DateFact(kind="ingested", value=date(2026, 1, 1), precision="day", source="system", confidence="high")  # type: ignore[arg-type]
    assert "ingested" not in VersionDates.model_fields


def test_dates_are_described_by_kind() -> None:
    text = describe_dates([fact("published", "2023-02-01"), fact("revised", "2024-01-01", "year", "pdf_info")])
    assert text == "published 2023-02-01 (source metadata); revised 2024 (PDF file info)"
    assert describe_dates([]) == "dates unknown"


def test_metadata_wins_over_weaker_sources_for_the_same_kind() -> None:
    weak = fact("published", "2021-01-01", source="pdf_info")
    strong = fact("published", "2023-02-01")
    assert describe_dates([weak, strong]).startswith("published 2023-02-01 (source metadata)")


# ----- in the graph --------------------------------------------------------------------------------------------------

V1_TEXT = "In version one the ranking model scores documents with BM25 term frequency weights and nothing else."
V2_TEXT = "In version two the ranking model scores documents with dense vectors instead of BM25 term weights."
CHUNKS = [
    chunk("c_v1", V1_TEXT, doc="doc_paper", version="ver_1", title="Ranking paper"),
    chunk("c_v2", V2_TEXT, doc="doc_paper", version="ver_2", title="Ranking paper"),
    replace(
        chunk(
            "c_undated",
            "An undated note says the ranking model scores documents with BM25.",
            doc="doc_note",
            version="ver_n",
        ),
        is_latest_version=None,
    ),
]


@dataclass
class Dates:
    by_document: dict[str, list[VersionDates]]

    def versions(self, workspace_id: str, document_ids: Sequence[str]) -> dict[str, list[VersionDates]]:
        return {d: self.by_document[d] for d in document_ids if d in self.by_document}


def paper(v2_fact: DateFact) -> Dates:
    return Dates(
        {
            "doc_paper": [
                version("ver_1", "v1", fact("published", "2023-02-01"), latest=False),
                version("ver_2", "v2", v2_fact, latest=True),
            ]
        }
    )


def analysis(*, as_of: str | None = None, wants_latest: bool = False) -> dict[str, Any]:
    return {
        "objective": "o",
        "subject": "s",
        "constraints": [],
        "expected_answer_type": "explanation",
        "temporal_scope": {"as_of": as_of, "wants_latest": wants_latest, "compare_versions": False},
        "needs_multi_doc": False,
        "single_hop": True,
    }


def draft_quoting_first_shown(request: LLMRequest) -> dict[str, Any]:
    evidence_id, text = next(iter(shown_evidence(request).items()))
    return {
        "claims": [
            {
                "id": "C1",
                "text": "The ranking model's scoring method.",
                "evidence_ids": [evidence_id],
                "quotes": [{"evidence_id": evidence_id, "text": text[:60]}],
                "kind": "fact",
            }
        ],
        "conflicts": [],
        "gaps": [],
    }


def run(llm: ScriptedLLM, dates: Dates, **state: Any) -> ResearchState:
    deps = WorkflowDeps(llm=llm, search=KeywordSearch(CHUNKS), temporal=dates)
    initial = ResearchState(
        run_id="run_T",
        workspace_id="ws_1",
        question="How does the ranking model score documents?",
        data_policy="any",
        budget=Budget(llm_calls_left=12, tokens_left=10**6, deadline=1e12),
        **state,
    )
    return run_to_end(build_graph(deps), initial)


COVERED = {"subquestions": [{"subquestion_id": "S1", "covered": True, "missing": ""}], "rationale": ""}


def answer_with_claim_markers(request: LLMRequest) -> str:
    claims = request.messages[-1].content.split("Verified claims:", 1)[1]
    markers = "".join(f"[{m}]" for m in dict.fromkeys(re.findall(r"\[(E\d+)\]", claims)))
    return f"The ranking model's scoring method is described in the cited version {markers}."


def script(**analysis_kwargs: Any) -> ScriptedLLM:
    return ScriptedLLM(
        {
            "query_classification": [analysis(**analysis_kwargs)],
            "evidence_synthesis": [draft_quoting_first_shown],
        },
        default={"evidence_assessment": COVERED, "final_response": answer_with_claim_markers},
    )


def synthesis_texts(llm: ScriptedLLM) -> list[str]:
    request = next(r for r in llm.requests if r.task_type == "evidence_synthesis")
    return list(shown_evidence(request).values())


def test_explicit_as_of_uses_only_the_version_that_applied_then() -> None:
    llm = script()
    final = run(llm, paper(fact("revised", "2024-03-15")), as_of=date(2023, 12, 31))
    shown = synthesis_texts(llm)
    assert V1_TEXT in shown and V2_TEXT not in shown
    assert final.answer is not None
    assert all(c.document_version_id == "ver_1" for c in final.answer.citations)
    # The undated note cannot support a claim about a date.
    assert any("no version dates are known" in line for line in final.answer.limitations)


def test_same_year_year_precision_version_is_listed_as_temporally_unresolved() -> None:
    llm = script()
    final = run(llm, paper(fact("revised", "2024-01-01", "year")), as_of=date(2024, 6, 1))
    assert V2_TEXT not in synthesis_texts(llm)
    assert final.answer is not None
    assert any("v2" in line and "temporally unresolved" in line for line in final.answer.limitations)


def test_as_of_read_from_the_question_is_disclosed() -> None:
    llm = script(as_of="2023-12-31")
    final = run(llm, paper(fact("revised", "2024-03-15")))
    assert V2_TEXT not in synthesis_texts(llm)
    assert final.answer is not None
    assert "The question was read as asking about the state on 2023-12-31." in final.answer.limitations


def test_latest_uses_the_latest_version_and_flags_unknown_latest() -> None:
    llm = script(wants_latest=True)
    final = run(llm, paper(fact("revised", "2024-03-15")))
    shown = synthesis_texts(llm)
    assert V2_TEXT in shown and V1_TEXT not in shown
    assert final.answer is not None
    assert any("which version is the latest is unknown" in line for line in final.answer.limitations)


def test_latest_filter_from_the_request_has_the_same_effect() -> None:
    llm = script()
    run(llm, paper(fact("revised", "2024-03-15")), filters=RunFilters(latest_only=True))
    assert V1_TEXT not in synthesis_texts(llm)


def test_answers_show_the_cited_versions_dates_by_kind() -> None:
    llm = script()
    final = run(llm, paper(fact("revised", "2024-03-15")), as_of=date(2023, 12, 31))
    answer = final.answer
    assert answer is not None and answer.citations
    assert answer.citations[0].dates == [fact("published", "2023-02-01")]
    assert "### Sources" in answer.markdown
    assert "Ranking paper (v1), p. 1: published 2023-02-01 (source metadata)" in answer.markdown
    synthesis = next(r for r in llm.requests if r.task_type == "evidence_synthesis")
    assert "dates: published 2023-02-01 (source metadata)" in synthesis.messages[-1].content
