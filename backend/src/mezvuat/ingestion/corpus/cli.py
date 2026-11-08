"""Corpus tooling: `mezvuat-corpus harvest | select | download`.

Downloaded PDFs are untrusted data: they are written to their own directory and only ever read as bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import Any

import httpx

from mezvuat.core.errors import ParseFailedError
from mezvuat.core.logging import configure_logging
from mezvuat.ingestion.corpus.manifest import manifest_from_records, select_papers
from mezvuat.ingestion.corpus.oai import (
    USER_AGENT,
    ArxivRecord,
    ArxivVersion,
    RateLimiter,
    fetch_record,
    search_ids,
)
from mezvuat.ingestion.corpus.splits import Family, build_splits
from mezvuat.ingestion.parsers.pdf import PdfParser

log = logging.getLogger("mezvuat.corpus")
MAX_PDF_BYTES = 40 * 1024 * 1024
DEFAULT_QUERY = (
    '(cat:cs.IR OR cat:cs.CL) AND (abs:"retrieval-augmented" OR abs:"retrieval augmented" OR abs:rerank '
    'OR abs:reranking OR abs:"dense retrieval" OR abs:"hybrid retrieval" OR abs:"passage retrieval" OR abs:"BM25")'
)


def _data_root() -> Path:
    return Path(os.environ.get("MEZVUAT_DATA_ROOT", "data")) / "corpus"


def _record_from_json(item: dict[str, Any]) -> ArxivRecord:
    return ArxivRecord(
        arxiv_id=item["arxiv_id"],
        title=item["title"],
        authors=item["authors"],
        categories=tuple(item["categories"]),
        license=item["license"],
        abstract=item["abstract"],
        versions=tuple(
            ArxivVersion(label=v["label"], ordinal=v["ordinal"], submitted=date.fromisoformat(v["submitted"]))
            for v in item["versions"]
        ),
    )


def cmd_harvest(args: argparse.Namespace) -> None:
    """Find candidates through the search API, then fetch licence and version data for each. Resumable."""
    out = _data_root() / "harvest.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out.exists():
        with out.open(encoding="utf-8") as handle:
            done = {json.loads(line)["arxiv_id"] for line in handle}
    limiter = RateLimiter()
    with (
        httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True) as client,
        out.open("a", encoding="utf-8") as handle,
    ):
        candidates = list(dict.fromkeys(search_ids(args.query, client, limiter, max_results=args.max_candidates)))
        log.info("candidates found", extra={"candidates": len(candidates), "already_done": len(done)})
        for number, arxiv_id in enumerate(candidates, start=1):
            if arxiv_id in done:
                continue
            record = fetch_record(arxiv_id, client, limiter)
            if record is not None:
                handle.write(json.dumps(record.to_json()) + "\n")
                handle.flush()
            if number % 100 == 0:
                log.info("harvest progress", extra={"processed": number})
    log.info("harvest complete", extra={"path": str(out)})


def cmd_select(args: argparse.Namespace) -> None:
    source = _data_root() / "harvest.jsonl"
    with source.open(encoding="utf-8") as handle:
        records = [_record_from_json(json.loads(line)) for line in handle]
    chosen = select_papers(records, target=args.target)
    manifest = manifest_from_records(chosen)
    path = Path(args.manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    versions = sum(len(r.versions) for r in chosen)
    multi = sum(1 for r in chosen if len(r.versions) >= 2)
    log.info(
        "manifest written",
        extra={
            "harvested": len(records),
            "papers": len(chosen),
            "versions": versions,
            "multi_version": multi,
        },
    )


def cmd_download(args: argparse.Namespace) -> None:
    path = Path(args.manifest)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    pdf_dir = _data_root() / "arxiv"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    limiter = RateLimiter()
    failures: list[str] = []
    with httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=120.0) as client:
        for paper in manifest["papers"][: args.limit or None]:
            for version in paper["versions"]:
                name = f"{paper['arxiv_id'].replace('/', '_')}{version['label']}.pdf"
                target = pdf_dir / name
                if target.exists():
                    version["raw_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
                    continue
                limiter.wait()
                try:
                    response = client.get(f"https://arxiv.org/pdf/{paper['arxiv_id']}{version['label']}")
                    response.raise_for_status()
                except httpx.HTTPError as exc:
                    failures.append(f"{name}: {type(exc).__name__}")
                    continue
                if len(response.content) > MAX_PDF_BYTES or not response.content.startswith(b"%PDF-"):
                    failures.append(f"{name}: rejected")
                    continue
                target.write_bytes(response.content)
                version["raw_sha256"] = hashlib.sha256(response.content).hexdigest()
            # Persist after each paper so an interrupted run loses at most one paper of work.
            path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info("download finished", extra={"failures": failures})


def cmd_split(args: argparse.Namespace) -> None:
    """Split families before any query exists. Text comes from each paper's latest downloaded version."""
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    pdf_dir = _data_root() / "arxiv"
    parser = PdfParser()
    families: list[Family] = []
    skipped: list[str] = []
    for paper in manifest["papers"]:
        downloaded = [v for v in paper["versions"] if v.get("raw_sha256")]
        if not downloaded:
            skipped.append(paper["arxiv_id"])
            continue
        latest = max(downloaded, key=lambda v: v["ordinal"])
        raw = (pdf_dir / f"{paper['arxiv_id'].replace('/', '_')}{latest['label']}.pdf").read_bytes()
        try:
            text = parser.parse(raw).text
        except ParseFailedError:
            skipped.append(paper["arxiv_id"])
            continue
        year = min(v["submitted"][:4] for v in paper["versions"])
        families.append(Family(paper["arxiv_id"], f"{paper['categories'][0]}/{year}", text))
    result = dict(build_splits(families, seed=args.seed))
    result["skipped"] = skipped
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    log.info("splits written", extra={"families": len(families), "skipped": len(skipped)})


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(prog="mezvuat-corpus")
    sub = parser.add_subparsers(required=True)
    p_harvest = sub.add_parser("harvest")
    p_harvest.add_argument("--query", default=DEFAULT_QUERY)
    p_harvest.add_argument("--max-candidates", type=int, default=1200)
    p_harvest.set_defaults(func=cmd_harvest)
    p_select = sub.add_parser("select")
    p_select.add_argument("--target", type=int, default=150)
    p_select.add_argument("--manifest", default="../data/manifests/demo_corpus.json")
    p_select.set_defaults(func=cmd_select)
    p_download = sub.add_parser("download")
    p_download.add_argument("--manifest", default="../data/manifests/demo_corpus.json")
    p_download.add_argument("--limit", type=int, default=0)
    p_download.set_defaults(func=cmd_download)
    p_split = sub.add_parser("split")
    p_split.add_argument("--manifest", default="../data/manifests/demo_corpus.json")
    p_split.add_argument("--out", default="../data/manifests/demo_corpus_splits.json")
    p_split.add_argument("--seed", type=int, default=13)
    p_split.set_defaults(func=cmd_split)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
