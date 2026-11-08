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

from mezvuat.core.logging import configure_logging
from mezvuat.ingestion.corpus.manifest import manifest_from_records, select_papers
from mezvuat.ingestion.corpus.oai import USER_AGENT, ArxivRecord, ArxivVersion, RateLimiter, harvest

log = logging.getLogger("mezvuat.corpus")
MAX_PDF_BYTES = 40 * 1024 * 1024


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
    out = _data_root() / "harvest.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    limiter = RateLimiter()
    count = 0
    with (
        httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True) as client,
        out.open("w", encoding="utf-8") as handle,
    ):
        for set_spec in args.sets:
            for record in harvest(set_spec, date.fromisoformat(args.since), client, limiter):
                handle.write(json.dumps(record.to_json()) + "\n")
                count += 1
                if count % 1000 == 0:
                    handle.flush()
                    log.info("harvested", extra={"records": count, "set": set_spec})
    log.info("harvest complete", extra={"records": count, "path": str(out)})


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


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(prog="mezvuat-corpus")
    sub = parser.add_subparsers(required=True)
    p_harvest = sub.add_parser("harvest")
    p_harvest.add_argument("--sets", nargs="+", default=["cs:cs:IR", "cs:cs:CL"])
    p_harvest.add_argument("--since", default="2023-06-01")
    p_harvest.set_defaults(func=cmd_harvest)
    p_select = sub.add_parser("select")
    p_select.add_argument("--target", type=int, default=150)
    p_select.add_argument("--manifest", default="../data/manifests/demo_corpus.json")
    p_select.set_defaults(func=cmd_select)
    p_download = sub.add_parser("download")
    p_download.add_argument("--manifest", default="../data/manifests/demo_corpus.json")
    p_download.add_argument("--limit", type=int, default=0)
    p_download.set_defaults(func=cmd_download)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
