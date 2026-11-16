"""NLI claim verification, the LLM-judge fallback and the conflicting-evidence section (ADR-0009, M5 part 3)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from mezvuat.citations import (
    EvidenceItem,
    NliScores,
    NliThresholds,
    SpanRef,
    load_thresholds,
    premise_window,
    sentence_bounds,
    verify_claims,
)
from mezvuat.inference import LLMRequest
from mezvuat.orchestration.nodes import NLI_UNAVAILABLE_LIMITATION, QUOTE_ONLY_LIMITATION
from tests.fakes import RuleNli, ScriptedLLM, id_for
from tests.unit.test_research_graph import analysis, answer_citing, coverage, run, state

SUPPORTED = "BM25 scores documents using term frequency and inverse document frequency."
CONTRADICTED = "BM25 ignores document length entirely."
LENGTH_QUOTE = "with length normalisation controlled by the parameter b"
TF_QUOTE = "It scores documents by term frequency and inverse document frequency"


# ----- premises and thresholds ---------------------------------------------------------------------------------------


def test_sentences_do_not_split_at_common_abbreviations() -> None:
    text = "Smith et al. report gains, e.g. on MS MARCO. Fig. 2 shows this. The end!"
    sentences = [text[a:b].strip() for a, b in sentence_bounds(text)]
    assert sentences == ["Smith et al. report gains, e.g. on MS MARCO.", "Fig. 2 shows this.", "The end!"]


def test_premise_is_the_quoted_sentence_with_one_neighbour_each_side() -> None:
    text = "One is first. Two holds the quote here. Three follows. Four is far away."
    start = text.index("the quote")
    premise, a, b = premise_window(text, start, start + len("the quote"))
    assert premise == "One is first. Two holds the quote here. Three follows."
    assert text[a:b].strip() == premise
    narrow, _, _ = premise_window(text, start, start + 9, sentences=0)
    assert narrow == "Two holds the quote here."


def test_long_premises_are_cut_around_the_quote() -> None:
    text = ("word " * 600).strip() + " the quoted bit " + ("more " * 600).strip()
    start = text.index("the quoted bit")
    premise, a, b = premise_window(text, start, start + 14, max_chars=300)
    assert "the quoted bit" in premise and b - a <= 300


def test_thresholds_label_and_tie_break() -> None:
    provisional = NliThresholds()
    assert not provisional.calibrated and "provisional" in provisional.source
    assert provisional.label(NliScores(entailment=0.6, contradiction=0.1, neutral=0.3)) == "supported"
    assert provisional.label(NliScores(entailment=0.1, contradiction=0.7, neutral=0.2)) == "contradicted"
    assert provisional.label(NliScores(entailment=0.4, contradiction=0.3, neutral=0.3)) == "not_enough_info"
    low = NliThresholds(entailment=0.3, contradiction=0.3, calibrated=True, source="test")
    assert low.label(NliScores(entailment=0.35, contradiction=0.45, neutral=0.2)) == "contradicted"
    assert low.label(NliScores(entailment=0.45, contradiction=0.45, neutral=0.1)) == "supported"


def test_thresholds_load_from_the_calibration_file_or_fall_back(tmp_path: Path) -> None:
    assert load_thresholds(tmp_path / "missing.json") == NliThresholds()
    assert load_thresholds(None) == NliThresholds()
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"entailment": 0.7, "contradiction": 0.6, "calibrated": True, "source": "x"}))
    loaded = load_thresholds(path)
    assert (loaded.entailment, loaded.contradiction, loaded.calibrated) == (0.7, 0.6, True)


def _item(evidence_id: str, text: str, document_id: str = "doc_1", char_start: int = 1000) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        chunk_id=f"c_{evidence_id}",
        document_id=document_id,
        document_version_id=f"v_{document_id}",
        document_title="T",
        version_label="v1",
        is_latest_version=True,
        text=text,
        char_start=char_start,
        char_end=char_start + len(text),
        page_start=1,
        page_end=1,
        section_path=[],
        section_kind="body",
        page_spans=[],
        retrieval_round=0,
        subquestion_id="S1",
        query="q",
        rank=1,
        score=1.0,
    )


def _span(item: EvidenceItem, quote: str) -> SpanRef:
    start = item.char_start + item.text.index(quote)
    end = start + len(quote)
    return SpanRef(evidence_id=item.evidence_id, quote=quote, status="ok", char_start=start, char_end=end)


def test_verify_claims_scores_every_span_in_one_call_and_joins_multi_span_premises() -> None:
    e1 = _item("E1", "Alpha uses sparse terms. It is fast. Nothing else.")
    e2 = _item("E2", "Beta uses dense vectors. It is slow.", document_id="doc_2")
    registry = {"E1": e1, "E2": e2}
    nli = RuleNli([("contradicts-me", "contradiction"), ("|| Alpha is sparse", "entailment")])
    verdicts = verify_claims(
        [
            ("Alpha is sparse.", [_span(e1, "Alpha uses sparse terms")]),
            ("Alpha and Beta differ.", [_span(e1, "Alpha uses sparse terms"), _span(e2, "Beta uses dense vectors")]),
            ("This contradicts-me.", [_span(e2, "Beta uses dense vectors")]),
        ],
        registry,
        nli,
        NliThresholds(),
    )
    assert len(nli.calls) == 1 and len(nli.calls[0]) == 5  # 1 + (2 spans + 1 joined) + 1
    assert nli.calls[0][0][0] == "Alpha uses sparse terms. It is fast."  # the quote widened to its sentences
    assert [v.label for v in verdicts] == ["supported", "not_enough_info", "contradicted"]
    assert verdicts[1].combined is not None and len(verdicts[1].spans) == 2
    assert all(v.model_id == "fake/nli" and v.thresholds_calibrated is False for v in verdicts)


def test_a_contradicting_span_outweighs_a_supporting_one() -> None:
    e1 = _item("E1", "Alpha is fast on CPU.")
    e2 = _item("E2", "Alpha is slow on CPU.", document_id="doc_2")
    nli = RuleNli([("slow", "contradiction"), ("fast", "entailment")])
    (verdict,) = verify_claims(
        [("Alpha runs quickly.", [_span(e1, "Alpha is fast"), _span(e2, "Alpha is slow")])],
        {"E1": e1, "E2": e2},
        nli,
        NliThresholds(),
    )
    assert verdict.label == "contradicted"
    assert [s.label for s in verdict.spans] == ["supported", "contradicted"]


def test_a_model_returning_the_wrong_number_of_scores_is_an_error() -> None:
    class Short(RuleNli):
        def predict(self, pairs: Any) -> list[NliScores]:
            return []

    e1 = _item("E1", "Alpha is fast on CPU.")
    with pytest.raises(ValueError, match="0 scores for 1 pairs"):
        verify_claims([("x", [_span(e1, "Alpha is fast")])], {"E1": e1}, Short(), NliThresholds())


# ----- the workflow --------------------------------------------------------------------------------------------------


def two_claims(request: LLMRequest) -> dict[str, Any]:
    bm25 = id_for(request, "BM25 is a bag-of-words")
    return {
        "claims": [
            {
                "id": "C1",
                "text": SUPPORTED,
                "evidence_ids": [bm25],
                "quotes": [{"evidence_id": bm25, "text": TF_QUOTE}],
                "kind": "fact",
            },
            {
                "id": "C2",
                "text": CONTRADICTED,
                "evidence_ids": [bm25],
                "quotes": [{"evidence_id": bm25, "text": LENGTH_QUOTE}],
                "kind": "fact",
            },
        ],
        "conflicts": [],
        "gaps": [],
    }


def one_claim(request: LLMRequest) -> dict[str, Any]:
    draft = two_claims(request)
    draft["claims"] = draft["claims"][:1]
    return draft


def llm(drafts: list[Any], **extra: list[Any]) -> ScriptedLLM:
    return ScriptedLLM(
        {
            "query_classification": [analysis(single_hop=True)],
            "evidence_assessment": [coverage(True)],
            "evidence_synthesis": drafts,
            "final_response": [answer_citing("BM25 scores documents")],
            **extra,
        }
    )


def section(markdown: str, title: str) -> str:
    body = markdown.split(f"### {title}\n\n", 1)[1]
    return body.split("\n\n### ", 1)[0]


def test_nli_supports_one_claim_and_a_contradicted_claim_goes_to_the_conflict_section() -> None:
    scripted = llm([two_claims, two_claims])
    nli = RuleNli([("ignores document length", "contradiction"), ("scores documents using", "entailment")])
    final, _ = run(scripted, nli=nli)
    answer = final.answer
    assert answer is not None
    by_id = {v.claim_id: v for v in answer.claims}
    assert (by_id["C1"].status, by_id["C1"].method) == ("supported", "nli")
    assert (by_id["C2"].status, by_id["C2"].method) == ("contradicted", "nli")
    assert by_id["C2"].nli is not None and by_id["C2"].nli.spans[0].scores.contradiction > 0.9
    # A contradicted claim gets the one re-synthesis, like an unsupported one, then is dropped.
    assert scripted.calls("evidence_synthesis") == 2
    assert scripted.calls("citation_verification") == 0  # the judge is off by default
    assert {c.claim_id for c in answer.citations} == {"C1"}
    (conflict,) = answer.conflicts
    assert conflict.claim_id == "C2" and conflict.method == "nli" and not conflict.between_documents
    (against,) = conflict.contradicted_by
    assert against.quote == LENGTH_QUOTE and against.page == 2 and against.document_id == "doc_bm25"
    conflicts_md = section(answer.markdown, "Conflicting evidence")
    assert CONTRADICTED in conflicts_md and f"Contradicted by evidence {against.evidence_id}" in conflicts_md
    assert "[E" not in conflicts_md  # not answer citations
    assert any("contradicts it (see Conflicting evidence)" in line for line in answer.limitations)
    assert any("NLI model (fake/nli)" in line and "provisional" in line for line in answer.limitations)
    assert QUOTE_ONLY_LIMITATION not in answer.limitations
    info = answer.verification
    assert info is not None and info.checks == ["quote_match", "nli"]
    assert info.nli_model == "fake/nli" and info.nli_thresholds_calibrated is False and not info.llm_judge_enabled
    assert info.claims_by_status == {"supported": 1, "contradicted": 1}


def test_sources_that_disagree_are_shown_with_both_citations_and_dates() -> None:
    def cross(request: LLMRequest) -> dict[str, Any]:
        bm25, rrf = id_for(request, "BM25 is a bag-of-words"), id_for(request, "Reciprocal rank fusion")
        return {
            "claims": [
                {
                    "id": "C1",
                    "text": "Ranking uses term statistics.",
                    "evidence_ids": [bm25, rrf],
                    "quotes": [
                        {"evidence_id": bm25, "text": TF_QUOTE},
                        {"evidence_id": rrf, "text": "combines ranked lists by summing the reciprocal"},
                    ],
                    "kind": "fact",
                }
            ],
            "conflicts": [],
            "gaps": [],
        }

    nli = RuleNli([("Reciprocal rank fusion", "contradiction"), ("bag-of-words", "entailment")])
    final, _ = run(llm([cross, cross]), nli=nli)
    assert final.answer is not None
    (conflict,) = final.answer.conflicts
    assert conflict.between_documents
    assert [c.document_id for c in conflict.supported_by] == ["doc_bm25"]
    assert [c.document_id for c in conflict.contradicted_by] == ["doc_rrf"]
    assert "The documents disagree on this point." in section(final.answer.markdown, "Conflicting evidence")
    assert "abstained" in final.answer.flags


def test_claims_nli_cannot_decide_are_kept_as_partially_supported() -> None:
    final, _ = run(llm([one_claim]), nli=RuleNli())
    answer = final.answer
    assert answer is not None
    (verdict,) = answer.claims
    assert (verdict.status, verdict.method) == ("partially_supported", "quote_match")
    assert verdict.nli is not None and verdict.nli.label == "not_enough_info"
    assert answer.citations and answer.rendering == "model"
    assert any("backed only by their verbatim quote" in line for line in answer.limitations)
    assert not answer.conflicts and "### Conflicting evidence" not in answer.markdown


def test_an_nli_failure_falls_back_to_the_quote_check_and_says_so(caplog: pytest.LogCaptureFixture) -> None:
    final, _ = run(llm([one_claim]), nli=RuleNli(fail=True))
    answer = final.answer
    assert answer is not None and final.status == "completed"
    (verdict,) = answer.claims
    assert (verdict.status, verdict.method, verdict.nli) == ("supported", "quote_match", None)
    assert NLI_UNAVAILABLE_LIMITATION in answer.limitations and QUOTE_ONLY_LIMITATION in answer.limitations
    assert answer.verification is not None and answer.verification.nli_unavailable
    assert "NLI verification failed" in caplog.text


def test_without_an_nli_model_the_m4_quote_check_stands() -> None:
    final, _ = run(llm([one_claim]))
    assert final.answer is not None
    (verdict,) = final.answer.claims
    assert (verdict.status, verdict.method) == ("supported", "quote_match")
    assert QUOTE_ONLY_LIMITATION in final.answer.limitations
    info = final.answer.verification
    assert info is not None and info.checks == ["quote_match"] and info.nli_model is None
    assert info.nli_thresholds_calibrated is None


# ----- the LLM judge (fakes only) ------------------------------------------------------------------------------------


def judge(verdict: str, quote: str | None = None) -> Any:
    def build(request: LLMRequest) -> dict[str, Any]:
        assert "Claim:" in request.messages[-1].content
        if verdict == "not_enough_info":
            return {"verdict": verdict, "evidence_id": "", "quote": "", "reason": "unclear"}
        return {
            "verdict": verdict,
            "evidence_id": id_for(request, "BM25 is a bag-of-words"),
            "quote": quote or TF_QUOTE,
            "reason": "scripted",
        }

    return build


def test_the_judge_decides_an_undecided_claim_when_its_quote_verifies() -> None:
    scripted = llm([one_claim], citation_verification=[judge("supported")])
    final, _ = run(scripted, nli=RuleNli(), llm_judge=True)
    assert final.answer is not None
    (verdict,) = final.answer.claims
    assert (verdict.status, verdict.method) == ("supported", "llm_judge")
    assert verdict.judge is not None and verdict.judge.accepted and verdict.judge.span is not None
    (request,) = [r for r in scripted.requests if r.task_type == "citation_verification"]
    assert request.data_policy == "any" and request.response_schema is not None and request.temperature == 0.0
    assert request.correlation.workflow_node == "verify"
    assert request.correlation.prompt_version.startswith("judge_claim@")
    assert [r.task_type for r in final.llm_calls].count("citation_verification") == 1
    assert any("LLM judge" in line for line in final.answer.limitations)
    assert final.answer.verification is not None and final.answer.verification.checks == [
        "quote_match",
        "nli",
        "llm_judge",
    ]


def test_a_judge_verdict_without_a_verbatim_quote_does_not_count() -> None:
    invented = judge("supported", quote="BM25 is the best ranking function ever")
    scripted = llm([one_claim], citation_verification=[invented])
    final, _ = run(scripted, nli=RuleNli(), llm_judge=True)
    assert final.answer is not None
    (verdict,) = final.answer.claims
    assert (verdict.status, verdict.method) == ("partially_supported", "quote_match")
    assert verdict.judge is not None and not verdict.judge.accepted
    assert any("quote" in p and "failed: not_found" in p for p in verdict.problems)


def test_a_judge_contradiction_is_listed_as_a_conflict_with_the_judges_span() -> None:
    scripted = ScriptedLLM(
        {
            "query_classification": [analysis(single_hop=True)],
            "evidence_assessment": [coverage(True)],
            "evidence_synthesis": [one_claim, one_claim],
        },
        default={"citation_verification": judge("contradicted", quote=LENGTH_QUOTE)},
    )
    final, _ = run(scripted, nli=RuleNli(), llm_judge=True)
    assert final.answer is not None
    (verdict,) = final.answer.claims
    assert (verdict.status, verdict.method) == ("contradicted", "llm_judge")
    (conflict,) = final.answer.conflicts
    assert conflict.method == "llm_judge" and conflict.contradicted_by[0].quote == LENGTH_QUOTE
    assert "(LLM judge)" in section(final.answer.markdown, "Conflicting evidence")
    assert "abstained" in final.answer.flags


def test_the_judge_never_spends_the_call_reserved_for_the_response() -> None:
    # understand, assess and synthesize leave one call: the judge needs two (its own and the response's).
    scripted = llm([one_claim], citation_verification=[judge("supported")])
    final, _ = run(scripted, initial=state(budget_calls=4), nli=RuleNli(), llm_judge=True)
    assert scripted.calls("citation_verification") == 0 and scripted.calls("final_response") == 1
    assert final.answer is not None and final.answer.claims[0].status == "partially_supported"


def test_judge_calls_are_capped_per_verification() -> None:
    scripted = llm([two_claims], citation_verification=[judge("not_enough_info")])
    final, _ = run(scripted, nli=RuleNli(), llm_judge=True, max_judge_calls=1)
    assert scripted.calls("citation_verification") == 1
    assert final.answer is not None
    assert [v.judge is not None for v in final.answer.claims] == [True, False]


# ----- the real model, on CPU (skipped unless it is in the local Hugging Face cache) ---------------------------------


def _model_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        from mezvuat.ml.registry import get_nli_spec
    except ImportError:
        return False
    spec = get_nli_spec("cross-encoder/nli-deberta-v3-small")
    return isinstance(try_to_load_from_cache(spec.model_id, "model.safetensors", revision=spec.revision), str)


@pytest.mark.live_model
@pytest.mark.skipif(not _model_cached(), reason="the NLI model is not in the local Hugging Face cache")
def test_the_pinned_model_runs_on_cpu_with_the_registered_label_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setitem(os.environ, "CUDA_VISIBLE_DEVICES", "-1")
    from mezvuat.ml.nli import CrossEncoderNli

    model = CrossEncoderNli.from_model_id("cross-encoder/nli-deberta-v3-small", device="cpu")
    assert model.device == "cpu"
    premise = "BM25 scores documents by term frequency and inverse document frequency."
    entail, contra, neutral = model.predict(
        [
            (premise, "BM25 uses term frequency."),
            (premise, "BM25 does not use term frequency."),
            (premise, "BM25 was invented in 2015 at a hackathon."),
        ]
    )
    thresholds = NliThresholds()
    assert thresholds.label(entail) == "supported"
    assert thresholds.label(contra) == "contradicted"
    assert neutral.entailment < 0.5
    assert abs(entail.entailment + entail.contradiction + entail.neutral - 1.0) < 1e-4
