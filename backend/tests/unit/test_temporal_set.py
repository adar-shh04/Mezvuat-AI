"""The temporal question set tooling (decision D11): drafting checks, date plans and the owner's review."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from mezvuat.evaluation import dataset_cli
from mezvuat.evaluation import temporal_set as ts
from mezvuat.evaluation.temporal_review import (
    REVIEW_FILE,
    TemporalReviewSession,
    load_items,
    run_temporal_review,
    temporal_status,
)
from mezvuat.temporal import DateFact, VersionDates, resolve_as_of

OLD = (
    "We train the retriever for three epochs on the full corpus. The reranker improves nDCG@10 from 0.41 to 0.47 "
    "on the validation split. Results are averaged over five seeds and reported with standard deviations."
)
NEW = (
    "We train the retriever for three epochs on the full corpus. The reranker improves nDCG@10 from 0.41 to 0.52 "
    "on the held-out validation split. Results are averaged over five seeds and reported with standard deviations."
)


def version(label: str, day: str, text: str) -> ts.VersionText:
    return ts.VersionText(label=label, submitted=date.fromisoformat(day), text=text, pages=[(1, 0, 10_000)])


def draft(**overrides: str) -> dict[str, Any]:
    base = {
        "candidate_id": "C1",
        "topic": "reranker gain",
        "as_of_question": "As of 2026-06-10, what nDCG@10 gain did 'Paper' report for the reranker?",
        "as_of_answer": "From 0.41 to 0.47.",
        "latest_question": "In the latest version of 'Paper', what nDCG@10 does the reranker reach?",
        "latest_answer": "0.52.",
        "change_question": "What changed in the reranker result of 'Paper' between v1 and v2?",
        "change_answer": "The reported nDCG@10 rose from 0.47 to 0.52, now on the held-out split.",
        "before_question": "As of 2026-05-05, what nDCG@10 gain did 'Paper' report?",
        "old_quote": "The reranker improves nDCG@10 from 0.41 to 0.47",
        "new_quote": "The reranker improves nDCG@10 from 0.41 to 0.52",
    }
    return {**base, **overrides}


def items_for(
    plan: ts.DatePlan, d: dict[str, Any] | None = None, undeterminable: bool = True
) -> tuple[list[dict[str, Any]], list[str]]:
    versions = [version("v1", "2026-06-04", OLD), version("v2", "2026-06-18", NEW)]
    return ts.build_items(
        arxiv_id="2606.00001",
        title="Paper",
        split="test",
        versions=versions,
        plan=plan,
        draft=d or draft(),
        include_undeterminable=undeterminable,
        drafted_by={"model": "fake"},
    )


def test_change_candidates_find_the_changed_number_first() -> None:
    candidates = ts.change_candidates(OLD, NEW)
    assert candidates and candidates[0].numbers_changed
    assert "0.47" in candidates[0].old and "0.52" in candidates[0].new
    assert ts.change_candidates(OLD, OLD) == []


def test_date_plan_uses_only_version_dates() -> None:
    plan = ts.plan_dates(date(2026, 6, 4), date(2026, 6, 18))
    assert plan.as_of == date(2026, 6, 11) and plan.same_day is None
    assert plan.before_first == date(2026, 5, 5)
    same = ts.plan_dates(date(2026, 6, 4), date(2026, 6, 4))
    assert same.as_of is None and same.same_day == date(2026, 6, 4)


def test_items_carry_verified_spans_and_expected_versions() -> None:
    items, problems = items_for(ts.plan_dates(date(2026, 6, 4), date(2026, 6, 18)))
    assert problems == []
    kinds = {i["kind"]: i for i in items}
    assert set(kinds) == {"as_of", "latest", "version_change", "undeterminable"}
    assert kinds["as_of"]["expected_versions"] == ["v1"] and kinds["latest"]["expected_versions"] == ["v2"]
    assert kinds["undeterminable"]["expected_behavior"] == "abstain"
    for item in items:
        for ev in item["evidence"]:
            text = OLD if ev["version"] == "v1" else NEW
            assert text[ev["char_start"] : ev["char_end"]] == ev["quote"]
            assert ev["page"] == 1


def test_expected_versions_agree_with_the_temporal_resolver() -> None:
    plan = ts.plan_dates(date(2026, 6, 4), date(2026, 6, 18))
    items, _ = items_for(plan)
    versions = [
        VersionDates(
            version_id=label,
            version_label=label,
            version_ordinal=n,
            is_latest_version=n == 2,
            facts=[DateFact(kind="published" if n == 1 else "revised", value=day, precision="day",
                            source="source_metadata", confidence="high")],
        )
        for n, (label, day) in enumerate([("v1", date(2026, 6, 4)), ("v2", date(2026, 6, 18))], start=1)
    ]  # fmt: skip
    for item in items:
        if item["as_of"] is None:
            continue
        resolved = resolve_as_of(versions, date.fromisoformat(item["as_of"]))
        expected = item["expected_versions"][0] if item["expected_behavior"] == "answer" else None
        assert resolved.applicable == expected


def test_same_day_versions_become_an_undeterminable_item() -> None:
    versions = [version("v1", "2026-06-04", OLD), version("v2", "2026-06-04", NEW)]
    items, _ = ts.build_items(
        arxiv_id="x", title="P", split="val", versions=versions, plan=ts.plan_dates(versions[0].submitted,
        versions[1].submitted), draft=draft(), include_undeterminable=False, drafted_by={},
    )  # fmt: skip
    kinds = [i["kind"] for i in items]
    assert "as_of" not in kinds and kinds.count("undeterminable") == 1
    assert next(i for i in items if i["kind"] == "undeterminable")["reason"] == "same_day_versions"


def test_unverifiable_quotes_drop_the_whole_draft() -> None:
    plan = ts.plan_dates(date(2026, 6, 4), date(2026, 6, 18))
    items, problems = items_for(plan, draft(old_quote="The reranker improves recall by a wide margin overall"))
    assert items == [] and "old_quote is not verbatim in the first version" in problems
    items, problems = items_for(plan, draft(new_quote=draft()["old_quote"]))
    assert items == []


def write_drafts(directory: Path, n: int = 10) -> None:
    plan = ts.plan_dates(date(2026, 6, 4), date(2026, 6, 18))
    base, _ = items_for(plan)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "drafts.jsonl").open("w", encoding="utf-8") as handle:
        for k in range(n):
            for item in base:
                handle.write(json.dumps({**item, "item_id": f"p{k}:{item['kind']}"}) + "\n")


def test_review_hides_about_a_fifth_and_records_to_its_own_file(tmp_path: Path) -> None:
    write_drafts(tmp_path)
    items = load_items(tmp_path, seed=13)
    assert len(items) == 40 and 5 <= sum(i.blind for i in items) <= 11
    session = TemporalReviewSession(items, tmp_path / REVIEW_FILE, reviewer="t")
    shown = next(i for i, it in enumerate(items) if not it.blind)
    blind = next(i for i, it in enumerate(items) if it.blind)
    session.decide(shown, "accept")
    with pytest.raises(ValueError):
        session.decide(blind, "accept")
    session.decide(blind, "write", answer="Cannot be determined.", behavior="abstain")
    session.undo()
    session.decide(blind, "reject")
    decided = session.decided()
    assert decided[items[shown].item_id]["agrees_with_draft_behavior"] is True
    assert decided[items[blind].item_id]["decision"] == "reject"
    assert "reviewed" in temporal_status(tmp_path, seed=13)


def test_review_loop_with_keys(tmp_path: Path) -> None:
    write_drafts(tmp_path, n=1)
    items = load_items(tmp_path, seed=13)
    keys = iter(["e" if not i.blind else "w" for i in items[:1]] + ["\n", "r", "q"])
    answers = iter(["Edited answer.", "a"])
    run_temporal_review(
        tmp_path, reviewer="t", seed=13, key=lambda: next(keys), prompt=lambda _: next(answers), clear=False
    )
    records = [json.loads(line) for line in (tmp_path / REVIEW_FILE).read_text(encoding="utf-8").splitlines()]
    assert records[0]["answer"] == "Edited answer." and records[0]["expected_behavior"] == "answer"


@pytest.mark.parametrize("split", ["val", "test", "temporal"])
def test_review_command_still_accepts_every_mode(split: str, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(dataset_cli, "cmd_review", lambda args: seen.append(args.split))
    monkeypatch.setattr(sys, "argv", ["mezvuat-dataset", "review", split])
    dataset_cli.main()
    assert seen == [split]


def test_val_and_test_review_paths_are_unchanged(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from mezvuat.evaluation import review

    called: dict[str, Any] = {}
    monkeypatch.setattr(review, "run_review", lambda split, **kw: called.update(split=split, **kw))
    monkeypatch.setattr(dataset_cli, "ensure_corpus", lambda: {})
    monkeypatch.setattr(sys, "argv", ["mezvuat-dataset", "review", "test"])
    dataset_cli.main()
    assert called["split"] == "test" and called["dataset_dir"] == dataset_cli.DATASET_DIR
    assert dataset_cli.TEMPORAL_DIR != dataset_cli.DATASET_DIR / "labels"


DRAFTS = dataset_cli.TEMPORAL_DIR / "drafts.jsonl"


@pytest.mark.skipif(not DRAFTS.exists(), reason="no temporal drafts committed yet")
def test_committed_drafts_agree_with_the_manifest_and_the_resolver() -> None:
    manifest = {p["arxiv_id"]: p for p in json.loads(dataset_cli.MANIFEST.read_text(encoding="utf-8"))["papers"]}
    rows = [json.loads(line) for line in DRAFTS.read_text(encoding="utf-8").splitlines()]
    assert len(rows) >= 30 and len({r["item_id"] for r in rows}) == len(rows)
    for row in rows:
        paper = manifest[row["arxiv_id"]]
        dates = {v["label"]: v["submitted"] for v in paper["versions"]}
        assert len(dates) >= 2, "only multi-version papers"
        assert {v["label"]: v["submitted"] for v in row["versions"]} == dates, "versions come from the manifest"
        for ev in row["evidence"]:
            assert ev["submitted"] == dates[ev["version"]] and 20 <= len(ev["quote"]) <= ts.MAX_QUOTE_CHARS
        if row["as_of"] is None:
            continue
        versions = [
            VersionDates(
                version_id=label, version_label=label, version_ordinal=n, is_latest_version=None,
                facts=[DateFact(kind="published", value=date.fromisoformat(day), precision="day",
                                source="source_metadata", confidence="high")],
            )
            for n, (label, day) in enumerate(sorted(dates.items()), start=1)
        ]  # fmt: skip
        applicable = resolve_as_of(versions, date.fromisoformat(row["as_of"])).applicable
        if row["expected_behavior"] == "abstain":
            assert applicable is None
        else:
            assert [applicable] == row["expected_versions"]


def test_questions_are_made_standalone_with_the_title() -> None:
    assert (
        ts.standalone_question("What did the paper report?", "A Title")
        == 'In the paper "A Title": What did the paper report?'
    )
    assert ts.standalone_question("What did A Title report?", "A Title") == "What did A Title report?"
    items, _ = items_for(ts.plan_dates(date(2026, 6, 4), date(2026, 6, 18)))
    assert all(i["question"].startswith('In the paper "Paper"') or "'Paper'" in i["question"] for i in items)
