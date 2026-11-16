"""NLI calibration tooling: the pair builder, the `review nli` mode and the threshold calibration (M5 part 3).

Calibration is tested on synthetic labels only. Nothing here is, or is reported as, a calibration of the real model.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from mezvuat.citations import NliScores, NliThresholds, premise_window
from mezvuat.evaluation import dataset_cli, nli_pairs
from mezvuat.evaluation import nli_calibration as cal
from mezvuat.evaluation.domain import DomainChunk
from mezvuat.evaluation.nli_review import (
    REVIEW_FILE,
    NliPair,
    NliReviewSession,
    load_pairs,
    nli_status,
    render,
    run_nli_review,
)

# ----- claim transforms -----------------------------------------------------------------------------------------------


def test_rewrites_drop_openers_frames_parentheticals_and_trailing_clauses() -> None:
    assert nli_pairs.rewrite("However, dense retrieval is effective for long and complex queries.") == (
        "Dense retrieval is effective for long and complex queries."
    )
    assert nli_pairs.rewrite("Our results show that dense retrieval is effective for long queries.") == (
        "Dense retrieval is effective for long queries."
    )
    assert nli_pairs.rewrite("The retriever (a BM25 variant) indexes every passage of the collection.") == (
        "The retriever indexes every passage of the collection."
    )
    assert nli_pairs.rewrite("The retriever indexes every passage of the corpus, which makes the index large.") == (
        "The retriever indexes every passage of the corpus."
    )
    assert nli_pairs.rewrite("The retriever indexes every passage of the whole collection.") is None


def test_number_perturbations_skip_years_and_references() -> None:
    rng = random.Random(1)  # noqa: S311
    changed = nli_pairs.perturb_number("In 2024 the model reached 61.4 nDCG on Table 3 of the suite.", rng)
    assert changed is not None
    text, detail = changed
    assert detail["from"] == "61.4" and detail["to"] != "61.4" and "2024" in text and "Table 3" in text
    assert nli_pairs.perturb_number("In 2024 we report the results in Table 3 and Section 4.", rng) is None


def test_negation_is_added_or_removed_but_never_doubled() -> None:
    assert nli_pairs.perturb_negation("The index is updated during evaluation runs.") == (
        "The index is not updated during evaluation runs.",
        {"from": "is", "to": "is not"},
    )
    assert nli_pairs.perturb_negation("The index is not updated during evaluation.") == (
        "The index is updated during evaluation.",
        {"from": "not", "to": ""},
    )
    assert nli_pairs.perturb_negation("No single model is optimal for every task here.") is None


def test_antonyms_and_swaps_change_direction() -> None:
    assert nli_pairs.perturb_antonym("Larger batches give higher throughput.") == (
        "Smaller batches give higher throughput.",
        {"from": "Larger", "to": "Smaller"},
    )
    swapped = nli_pairs.perturb_swap("ColBERT outperforms BM25 on every dataset we tried.")
    assert swapped is not None and swapped[0] == "BM25 outperforms ColBERT on every dataset we tried."
    assert nli_pairs.perturb_swap("ColBERT and BM25 are both used in this study.") is None


def test_sentence_filter_rejects_references_formulas_and_fragments() -> None:
    assert nli_pairs.good_sentence("Dense retrieval encodes queries and passages into vectors for ranking them.")
    for bad in (
        "Yoon, S.; Choi, E.; and Kim, J. 2024. ListT5: Listwise Reranking Improves Zero-shot Retrieval.",
        "The reward is R2 = R1 + MRR + NDCG for every sequence that the policy generates here.",
        "dense retrieval encodes queries and passages into vectors for ranking them all.",
        "Table 3 reports the results of every configuration that we evaluated in the study.",
    ):
        assert not nli_pairs.good_sentence(bad), bad


# ----- skip rules (v2): one test per rule, with the cases found in the review of v1 -----------------------------------


def _number(sentence: str) -> tuple[Any, Counter[str]]:
    stats: Counter[str] = Counter()
    return nli_pairs.perturb_number(sentence, random.Random(7), stats), stats  # noqa: S311


def test_numbers_inside_citation_brackets_are_never_changed() -> None:
    result, stats = _number("Prior work [22,19,9,8] and later studies [13, 23, 61] use dense retrieval at scale.")
    assert result is None and stats["number:inside_citation_brackets"] == 3


def test_versions_licences_dataset_names_and_venues_are_not_quantities() -> None:
    for sentence, rule in (
        ("The code is released under the Apache 2.0 licence for everyone.", "number:version_licence_dataset_or_venue"),
        ("We evaluate the reader on SQuAD 2.0 and report the exact match.", "number:version_licence_dataset_or_venue"),
        ("We evaluate the reader on TriviaQA 1.0 and report the exact match.", "number:after_a_name"),
        ("We use version 3 of the toolkit for all the experiments here.", "number:version_licence_dataset_or_venue"),
    ):
        result, stats = _number(sentence)
        assert result is None and stats[rule] == 1, sentence
    # A venue year written with an apostrophe is not even a candidate.
    assert _number("This paper appears at CIKM \u201926 in the applied research track.")[0] is None
    assert _number("This paper appears at CIKM '26 in the applied research track.")[0] is None


def test_grouped_numbers_are_changed_whole_or_not_at_all() -> None:
    result, _ = _number("The corpus holds 14,692,232 passages after deduplication.")
    assert result is not None
    text, detail = result
    assert detail["from"] == "14,692,232" and "," in detail["to"] and detail["to"] in text
    assert "14,692,232" not in text and "7," not in text.replace(detail["to"], "")


def test_years_ranges_ratios_and_unmeasured_numbers_are_skipped() -> None:
    for sentence, rule in (
        ("In 2024 the community moved on to retrieval-augmented generation.", "number:year"),
        ("Each query requires reasoning over 2\u20134 documents in the collection.", "number:end_of_a_dash_range"),
        (
            "The pipeline has 3 stages that run one after another on the server.",
            "number:not_a_measurement_or_comparison",
        ),
    ):
        result, stats = _number(sentence)
        assert result is None and stats[rule] >= 1, sentence
    assert _number("PRR improves or matches the baseline in 94/120 conditions overall.")[0] is None
    assert _number("Linear interpolation merges all parameters with weights 2:1:1 in our runs.")[0] is None


def test_measured_numbers_are_preferred_over_other_numbers() -> None:
    result, _ = _number("Across 3 datasets the reranker gains 4.1 points of nDCG over BM25.")
    assert result is not None and result[1]["from"] == "4.1"


def test_deictic_and_positional_words_are_not_antonyms() -> None:
    stats: Counter[str] = Counter()
    assert nli_pairs.perturb_antonym("The pipeline described above runs on one machine.", stats) is None
    assert stats["antonym:deictic_or_positional_word"] == 1
    swapped = nli_pairs.perturb_antonym("As described above, recall is higher with fusion.")
    assert swapped is not None and swapped[1] == {"from": "higher", "to": "lower"}


def test_fixed_terms_idioms_and_discourse_markers_are_not_antonyms() -> None:
    for sentence in (
        "Retrieval augmented generation helps large language models answer questions.",
        "The updated synthesis no longer described the impacts in chemical terms.",
        "More broadly, the study highlights the importance of retrieval precision.",
        "To better exploit the training signal, we add a joint reranker.",
        "We ask how many escalators the building has in the example query.",
        "The local result should be read as a lower-level systems component.",
    ):
        stats: Counter[str] = Counter()
        assert nli_pairs.perturb_antonym(sentence, stats) is None, sentence
        assert stats["antonym:fixed_term_idiom_or_discourse_marker"] >= 1, sentence


def test_swaps_need_two_distinct_systems_around_a_comparison() -> None:
    for sentence, rule in (
        ("REACT-BM25 outperforms REACT on every benchmark in the suite.", "swap:names_contain_each_other"),
        ("ColBERT and BM25 are both used throughout this study.", "swap:no_comparative_relation"),
        ("A scorer fine-tuned on NASA- EOBench beats cosine and BM25 baselines.", "swap:first_name_is_not_the_subject"),
        ("BGE scores well overall; the comparison with ColBERT is in the appendix.", "swap:no_comparative_relation"),
    ):
        stats: Counter[str] = Counter()
        assert nli_pairs.perturb_swap(sentence, stats) is None, sentence
        assert stats[rule] == 1, sentence
    # A metric is what is compared, not a system: "F1 exceeds ... PPR" must not become "PPR exceeds ... F1".
    assert nli_pairs.perturb_swap("Its average F1 exceeds those of PPR and beam search by 0.8 points.") is None
    assert nli_pairs.perturb_swap("NEST achieves lower latency than CHyD on QuoteSum.") == (
        "CHyD achieves lower latency than NEST on QuoteSum.",
        {"from": "NEST ... CHyD", "to": "CHyD ... NEST"},
    )


def test_negation_skips_words_that_already_negate_and_main_verbs() -> None:
    for sentence, rule in (
        (
            "The model expresses uncertainty when evidence is insufficient for an answer.",
            "negation:negative_word_at_insertion_point",
        ),
        (
            "Some facts of a case may be irrelevant to some of its questions.",
            "negation:negative_word_at_insertion_point",
        ),
        (
            "Data discovery is less a problem of scarcity than of fragmented metadata.",
            "negation:negative_word_at_insertion_point",
        ),
        (
            "Five ArguAna queries have judged-positive documents missing from the archive.",
            "negation:have_or_do_as_main_verb",
        ),
        (
            "What these paradigms lack is an explicit evaluator for candidate sets.",
            "negation:sentence_already_negative",
        ),
    ):
        stats: Counter[str] = Counter()
        assert nli_pairs.perturb_negation(sentence, stats) is None, sentence
        assert stats[rule] == 1, sentence
    assert nli_pairs.perturb_negation("Software development has been central to the community.") == (
        "Software development has not been central to the community.",
        {"from": "has", "to": "has not"},
    )


def test_glued_text_and_prompt_templates_never_reach_labelling() -> None:
    glued = "The two core stages are retrievalandgeneration:theretrieverrunsavectorsearch over the corpus."
    assert not nli_pairs.good_sentence(glued)
    assert not nli_pairs.good_sentence("You are an expert relevance judge for conversational search tasks.")
    chunk = corpus(1)[0]
    broken = DomainChunk(**{**chunk.__dict__, "chunk_id": "glued", "text": chunk.text + " " + glued})
    assert [c.chunk_id for c in nli_pairs.eligible_chunks([chunk, broken])] == [chunk.chunk_id]
    assert len(nli_pairs.eligible_chunks([chunk, broken], exclude_glued=False)) == 2


def test_the_audit_counts_rules_over_the_whole_pool() -> None:
    audit = nli_pairs.rejection_audit(corpus(), seed=13)
    assert audit["candidate_sentences"] > 0 and audit["glued_text_chunks_excluded"] == 0
    assert audit["perturbations_possible"]["perturb_number"] > 0
    assert all(":" in reason for reason in audit["rejections"])


# ----- the pair builder -----------------------------------------------------------------------------------------------

TEMPLATES = (
    "The proposed retriever R{p} achieves a recall of {n}.5 points on the benchmark collection.",
    "Our experiments show that dense retrieval is effective for long queries in domain {p}.",
    "However, sparse methods remain strong baselines for keyword queries in setting {c}.",
    "System Alpha{p}X outperforms Beta{p}Y on every dataset that we evaluated here.",
    "The index for corpus {p} is built once and is updated during the evaluation phase.",
    "Larger batches lead to higher throughput when the model runs on machine {c}.",
    "We release the code and the trained checkpoints to support future work on topic {p}.",
    "Annotators judged each passage of batch {c} independently, which reduces anchoring.",
)


def corpus(papers: int = 20, split: str = "val") -> list[DomainChunk]:
    chunks = []
    for p in range(papers):
        for c in range(3):
            text = " ".join(t.format(p=p, c=c, n=40 + p) for t in TEMPLATES)
            chunks.append(
                DomainChunk(
                    chunk_id=f"{split}{p:02d}c{c}",
                    arxiv_id=f"2606.{p:05d}",
                    version_label="v1",
                    split=split,
                    title=f"Paper {p}",
                    section_path=(f"{c + 1} Section",),
                    section_kind="body",
                    page_start=c + 1,
                    page_end=c + 1,
                    token_count=120,
                    text=text,
                    text_sha256=f"h{p}{c}",
                )
            )
    return chunks


def test_pairs_are_balanced_deterministic_and_capped_per_paper() -> None:
    chunks = corpus() + corpus(5, split="test")
    rows = nli_pairs.build_pairs(chunks, seed=13)
    assert rows == nli_pairs.build_pairs(chunks, seed=13)
    assert [r["pair_id"] for r in rows] == [f"nli_{n:03d}" for n in range(1, len(rows) + 1)]
    assert 100 <= len(rows) <= nli_pairs.TARGET
    expected = Counter(r["construction"]["expected"] for r in rows)
    assert max(expected.values()) - min(expected.values()) <= 0.2 * len(rows)
    papers = Counter(r["claim_source"]["arxiv_id"] for r in rows)
    assert max(papers.values()) <= nli_pairs.MAX_PAIRS_PER_PAPER
    assert all(r["premise"]["arxiv_id"].startswith("2606.") for r in rows)
    assert not any(r["claim_source"]["chunk_id"].startswith("test") for r in rows), "val families only"
    for row in rows:
        method = row["construction"]["method"]
        assert row["construction"]["expected"] == nli_pairs.QUOTAS[method][0]
        source = row["claim_source"]["sentence"]
        if method.startswith(("sentence_", "perturb_")):
            assert source in row["premise"]["text"]
        else:
            assert row["claim"] not in row["premise"]["text"]
        if method.startswith("perturb_"):
            assert row["claim"] != source


def test_the_premise_is_the_verifiers_window() -> None:
    chunks = {c.chunk_id: c for c in corpus()}
    for row in nli_pairs.build_pairs(chunks.values(), seed=13)[:30]:
        premise = row["premise"]
        chunk = chunks[premise["chunk_id"]]
        source_start = chunk.text.find(row["claim_source"]["sentence"])
        if row["claim_source"]["chunk_id"] == premise["chunk_id"] and row["construction"]["method"] != (
            "same_passage_far"
        ):
            window = premise_window(chunk.text, source_start, source_start + len(row["claim_source"]["sentence"]))
            assert window[0] == premise["text"]


COMMITTED = dataset_cli.NLI_DIR / "pairs.jsonl"


@pytest.mark.skipif(not COMMITTED.exists(), reason="no NLI pairs committed")
def test_committed_pairs_match_their_manifest_and_come_from_val_families() -> None:
    rows = [json.loads(line) for line in COMMITTED.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads((dataset_cli.NLI_DIR / "manifest.json").read_text(encoding="utf-8"))
    splits = json.loads(dataset_cli.SPLITS.read_text(encoding="utf-8"))["assignment"]
    assert len(rows) == manifest["summary"]["pairs"] == 150
    assert len({r["pair_id"] for r in rows}) == len(rows)
    assert dict(Counter(r["construction"]["method"] for r in rows)) == manifest["summary"]["by_method"]
    assert dict(Counter(r["construction"]["expected"] for r in rows)) == {
        "supported": 50,
        "contradicted": 50,
        "neutral": 50,
    }
    for row in rows:
        assert splits[row["premise"]["arxiv_id"]] == "val" and splits[row["claim_source"]["arxiv_id"]] == "val"
        assert row["claim"].strip() and len(row["premise"]["text"]) <= 1500


# ----- review nli -----------------------------------------------------------------------------------------------------


def write_pairs(directory: Path, n: int = 6) -> list[NliPair]:
    directory.mkdir(parents=True, exist_ok=True)
    rows = nli_pairs.build_pairs(corpus(), seed=13)[:n]
    (directory / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return load_pairs(directory)


def test_the_review_screen_shows_no_suggestion_of_any_kind(tmp_path: Path) -> None:
    for pair in write_pairs(tmp_path):
        screen = render(pair, done=0, total=6)
        assert pair.data["claim"][:40] in screen.replace("\n", " ")
        construction = pair.data["construction"]
        for hidden in (construction["method"], "expected", "construction", "perturb", "entail", "score", "NLI"):
            assert hidden not in screen
        for value in construction["detail"].values():
            assert f"from: {value}" not in screen


def test_review_saves_every_label_resumes_and_undoes(tmp_path: Path) -> None:
    pairs = write_pairs(tmp_path)
    path = tmp_path / REVIEW_FILE
    session = NliReviewSession(pairs, path, reviewer="t")
    session.label(0, "supported")
    session.label(1, "contradicted")
    session.undo()
    resumed = NliReviewSession(pairs, path, reviewer="t")
    assert set(resumed.labelled()) == {pairs[0].pair_id}
    assert resumed.next_index() == 1
    record = resumed.labelled()[pairs[0].pair_id]
    assert record["suggestion_shown"] is False and record["fingerprint"] == pairs[0].fingerprint
    with pytest.raises(ValueError):
        session.label(2, "entailment")


def test_a_label_for_changed_pair_text_does_not_count(tmp_path: Path) -> None:
    pairs = write_pairs(tmp_path)
    NliReviewSession(pairs, tmp_path / REVIEW_FILE, reviewer="t").label(0, "neutral")
    edited = [NliPair({**pairs[0].data, "claim": "Something else entirely."}), *pairs[1:]]
    assert NliReviewSession(edited, tmp_path / REVIEW_FILE, reviewer="t").labelled() == {}


def test_review_loop_with_keys_and_status(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pairs = write_pairs(tmp_path, n=4)
    keys = iter(["y", "?", "c", "s", "x", "z", "n", "q"])
    run_nli_review(tmp_path, reviewer="t", key=lambda: next(keys), clear=False)
    records = [json.loads(line) for line in (tmp_path / REVIEW_FILE).read_text(encoding="utf-8").splitlines()]
    assert [r["label"] for r in records] == ["supported", "contradicted", "unusable", "neutral"]
    assert [r["pair_id"] for r in records] == [pairs[0].pair_id, pairs[1].pair_id, pairs[3].pair_id, pairs[2].pair_id]
    assert "All 4 NLI pairs are labelled" in capsys.readouterr().out
    assert "4 labelled" in nli_status(tmp_path)


@pytest.mark.parametrize("split", ["val", "test", "temporal", "nli"])
def test_review_command_accepts_every_mode(split: str, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(dataset_cli, "cmd_review", lambda args: seen.append(args.split))
    monkeypatch.setattr(sys, "argv", ["mezvuat-dataset", "review", split])
    dataset_cli.main()
    assert seen == [split]


@pytest.mark.parametrize("split", ["val", "test", "temporal", "nli"])
def test_each_review_mode_reaches_its_own_tool_and_file(split: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from mezvuat.evaluation import nli_review, review, temporal_review

    called: dict[str, Any] = {}
    monkeypatch.setattr(review, "run_review", lambda split, **kw: called.update(tool="relevance", split=split, **kw))
    monkeypatch.setattr(
        temporal_review, "run_temporal_review", lambda d, **kw: called.update(tool="temporal", directory=d)
    )
    monkeypatch.setattr(nli_review, "run_nli_review", lambda d, **kw: called.update(tool="nli", directory=d))
    monkeypatch.setattr(dataset_cli, "ensure_corpus", lambda: {})
    monkeypatch.setattr(sys, "argv", ["mezvuat-dataset", "review", split])
    dataset_cli.main()
    if split in ("val", "test"):
        assert called["tool"] == "relevance" and called["split"] == split
        assert called["dataset_dir"] == dataset_cli.DATASET_DIR and called["hide_all"] is True
    elif split == "temporal":
        assert called == {"tool": "temporal", "directory": dataset_cli.TEMPORAL_DIR}
    else:
        assert called == {"tool": "nli", "directory": dataset_cli.NLI_DIR}
        assert dataset_cli.NLI_DIR not in (dataset_cli.DATASET_DIR / "labels", dataset_cli.TEMPORAL_DIR)


def test_review_scripts_offer_the_nli_mode() -> None:
    scripts = dataset_cli.REPO / "scripts"
    assert '"val", "test", "temporal", "nli"' in (scripts / "review.ps1").read_text(encoding="utf-8")


# ----- calibration (synthetic labels only) ----------------------------------------------------------------------------


def synthetic(n_per_label: int = 40, seed: int = 0, noise: float = 0.0) -> list[cal.Example]:
    """Scores drawn around each label, with some overlap; a stand-in for labelled pairs."""
    rng = random.Random(seed)  # noqa: S311
    examples = []
    centres = {"supported": (0.8, 0.05), "contradicted": (0.05, 0.75), "not_enough_info": (0.25, 0.2)}
    for label, (e, c) in centres.items():
        for i in range(n_per_label):
            pe = min(max(e + rng.gauss(0, 0.12 + noise), 0.0), 1.0)
            pc = min(max(c + rng.gauss(0, 0.12 + noise), 0.0), 1.0 - pe)
            scores = NliScores(entailment=pe, contradiction=pc, neutral=1 - pe - pc)
            examples.append(cal.Example(f"{label}_{i}", scores, label))  # type: ignore[arg-type]
    return examples


def test_the_fit_beats_the_provisional_rule_and_meets_the_precision_target() -> None:
    examples = synthetic()
    fit = cal.fit_thresholds(examples, min_supported_precision=0.9)
    assert fit.thresholds.calibrated and fit.precision_target_met
    assert fit.evaluation.precision("supported") >= 0.9
    assert fit.evaluation.macro_f1 >= cal.evaluate(examples, NliThresholds()).macro_f1
    assert 0.1 <= fit.thresholds.entailment <= 0.95 and 0.1 <= fit.thresholds.contradiction <= 0.95


def test_a_stricter_precision_target_raises_the_entailment_threshold() -> None:
    examples = synthetic(noise=0.08)
    loose = cal.fit_thresholds(examples, min_supported_precision=None)
    strict = cal.fit_thresholds(examples, min_supported_precision=0.99)
    assert strict.thresholds.entailment >= loose.thresholds.entailment
    assert strict.evaluation.precision("supported") >= loose.evaluation.precision("supported")


def test_evaluation_counts_a_known_confusion() -> None:
    result = cal.score_predictions(
        [("supported", "supported"), ("supported", "not_enough_info"), ("contradicted", "contradicted")]
    )
    assert result.n == 3 and result.accuracy == pytest.approx(2 / 3)
    assert result.per_label["supported"]["recall"] == 0.5 and result.per_label["supported"]["precision"] == 1.0
    assert result.confusion["supported"]["not_enough_info"] == 1


def test_cross_validation_holds_out_every_example_once() -> None:
    examples = synthetic(n_per_label=25)
    report = cal.cross_validate(examples, k=5, seed=3)
    assert report["held_out"]["n"] == len(examples) and len(report["thresholds_per_fold"]) == 5
    folds = cal.stratified_folds(examples, 5, 3)
    assert sorted(e.pair_id for fold in folds for e in fold) == sorted(e.pair_id for e in examples)
    assert all(Counter(e.gold for e in fold)["supported"] == 5 for fold in folds)


def test_calibration_refuses_too_few_labels_and_skips_unusable_pairs() -> None:
    with pytest.raises(cal.InsufficientLabelsError, match="at least 20"):
        cal.calibrate(synthetic(n_per_label=10), source="synthetic")
    labels = {"a": {"label": "supported"}, "b": {"label": "unusable"}, "c": {"label": "neutral"}}
    scores = {"a": NliScores(entailment=0.9, contradiction=0.05, neutral=0.05)}
    examples, skipped = cal.examples_from(labels, scores)
    assert [e.gold for e in examples] == ["supported"] and skipped == {"unusable": 1, "no_score": 1}
    report = cal.calibrate(synthetic(), source="synthetic test data")
    assert report["calibrated"] is True and report["source"] == "synthetic test data"
    assert report["cross_validation"]["held_out"]["n"] == 120


def test_the_calibrate_command_refuses_to_run_without_owner_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_pairs(tmp_path)
    monkeypatch.setattr(dataset_cli, "NLI_DIR", tmp_path)
    monkeypatch.setitem(sys.modules, "mezvuat.ml.nli", None)  # importing the model would fail the test
    monkeypatch.setattr(sys, "argv", ["mezvuat-dataset", "nli-calibrate"])
    with pytest.raises(SystemExit, match="No owner labels yet"):
        dataset_cli.main()
    session = NliReviewSession(load_pairs(tmp_path), tmp_path / REVIEW_FILE, reviewer="t")
    session.label(0, "supported")
    with pytest.raises(SystemExit, match="Nothing was calibrated"):
        dataset_cli.main()
    assert not (tmp_path / "thresholds.json").exists()
