"""ADR-0003 spike (decision D6): compare PDF text extractors on the demo corpus.

Reference text is the arXiv-rendered HTML of the same version, which is independent of any PDF extractor. Metrics:
- reference_5gram_recall: share of reference word 5-grams found in the extracted text. Sensitive to reading order,
  garbled columns and dropped text.
- extracted_5gram_precision: share of extracted 5-grams found in the reference. Sensitive to interleaved junk.
- abstract_5gram_recall: the same recall, against the abstract in the arXiv metadata.

PDFs and HTML are untrusted input; they are only parsed for text. Usage (from backend/):
  uv run --no-sync python ../experiments/spikes/pdf_extractors.py --papers 30 --out ../docs/eval/pdf_extractors.json
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import time
from collections.abc import Callable
from pathlib import Path

import httpx
import numpy as np
import pdfplumber
import pymupdf
import trafilatura

from mezvuat.evaluation.metrics import bootstrap_ci
from mezvuat.ingestion.corpus.oai import USER_AGENT, RateLimiter

_WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    # Rejoin words hyphenated across line ends so hyphenation does not count against any extractor.
    return _WORD.findall(re.sub(r"-\s*\n\s*", "", text).lower())


def grams(tokens: list[str], n: int = 5) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def pymupdf_native(raw: bytes) -> list[str]:
    with pymupdf.open(stream=raw, filetype="pdf") as document:
        return [str(page.get_text("text")) for page in document]


def pymupdf_blocks(raw: bytes) -> list[str]:
    pages: list[str] = []
    with pymupdf.open(stream=raw, filetype="pdf") as document:
        for page in document:
            blocks = page.get_text("blocks", sort=False)
            pages.append("\n\n".join(str(b[4]) for b in blocks if b[6] == 0))
    return pages


def pdfplumber_default(raw: bytes) -> list[str]:
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        return [page.extract_text() or "" for page in pdf.pages]


def _column_split(page: pdfplumber.page.Page) -> float | None:
    """Find a vertical gutter near the page centre: a band that almost no characters cross."""
    width = float(page.width)
    chars = page.chars
    if len(chars) < 200:
        return None
    bins = np.zeros(int(width) + 1)
    for char in chars:
        bins[int(char["x0"]) : int(char["x1"]) + 1] += 1
    centre_low, centre_high = int(width * 0.4), int(width * 0.6)
    band = bins[centre_low:centre_high]
    lowest = int(band.argmin()) + centre_low
    busy = float(np.percentile(bins[bins > 0], 50))
    return float(lowest) if bins[lowest] < 0.05 * busy else None


def pdfplumber_columns(raw: bytes) -> list[str]:
    """A fair pdfplumber: crop two-column pages at the detected gutter and read left then right."""
    output: list[str] = []
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        for page in pdf.pages:
            split = _column_split(page)
            if split is None:
                output.append(page.extract_text() or "")
                continue
            left = page.crop((0, 0, split, page.height)).extract_text() or ""
            right = page.crop((split, 0, page.width, page.height)).extract_text() or ""
            output.append(f"{left}\n\n{right}")
    return output


EXTRACTORS: dict[str, Callable[[bytes], list[str]]] = {
    "pymupdf_text": pymupdf_native,
    "pymupdf_blocks": pymupdf_blocks,
    "pdfplumber_default": pdfplumber_default,
    "pdfplumber_columns": pdfplumber_columns,
}


def fetch_reference(client: httpx.Client, limiter: RateLimiter, arxiv_id: str, label: str, cache: Path) -> str | None:
    path = cache / f"{arxiv_id}{label}.txt"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        return text or None
    limiter.wait()
    response = client.get(f"https://arxiv.org/html/{arxiv_id}{label}")
    extracted = (
        trafilatura.extract(response.text, include_comments=False, include_tables=False)
        if response.status_code == 200 and "text/html" in response.headers.get("content-type", "")
        else None
    )
    path.write_text(extracted or "", encoding="utf-8")
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("../data/manifests/demo_corpus.json"))
    parser.add_argument("--papers", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = Path(os.environ.get("MEZVUAT_DATA_ROOT", "data")) / "corpus"
    cache = root / "html_reference"
    cache.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    limiter = RateLimiter()
    results: list[dict[str, object]] = []
    with httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=60.0) as client:
        for paper in manifest["papers"]:
            if len(results) >= args.papers:
                break
            version = paper["versions"][-1]
            path = root / "arxiv" / f"{paper['arxiv_id']}{version['label']}.pdf"
            if not path.exists():
                continue
            reference = fetch_reference(client, limiter, paper["arxiv_id"], version["label"], cache)
            if reference is None or len(reference) < 2000:
                continue
            raw = path.read_bytes()
            abstract = grams(words(paper["abstract"]))
            ref_grams = grams(words(reference))
            row: dict[str, object] = {"arxiv_id": paper["arxiv_id"], "label": version["label"]}
            for name, extractor in EXTRACTORS.items():
                start = time.perf_counter()
                pages = extractor(raw)
                elapsed = time.perf_counter() - start
                extracted = grams(words("\n\n".join(pages)))
                row[name] = {
                    "seconds_per_page": elapsed / max(len(pages), 1),
                    "abstract_5gram_recall": len(abstract & extracted) / max(len(abstract), 1),
                    "reference_5gram_recall": len(ref_grams & extracted) / max(len(ref_grams), 1),
                    "extracted_5gram_precision": len(ref_grams & extracted) / max(len(extracted), 1),
                }
            results.append(row)

    summary: dict[str, object] = {"n_papers": len(results)}
    for name in EXTRACTORS:
        for metric in (
            "seconds_per_page",
            "abstract_5gram_recall",
            "reference_5gram_recall",
            "extracted_5gram_precision",
        ):
            values = [float(r[name][metric]) for r in results]  # type: ignore[index]
            low, high = bootstrap_ci(values)
            summary[f"{name}.{metric}"] = {"mean": float(np.mean(values)), "ci95": [low, high]}
    baseline = "pdfplumber_columns"
    for name in ("pymupdf_text", "pymupdf_blocks", "pdfplumber_default"):
        for metric in ("reference_5gram_recall", "extracted_5gram_precision"):
            diffs = [float(r[name][metric]) - float(r[baseline][metric]) for r in results]  # type: ignore[index]
            low, high = bootstrap_ci(diffs)
            summary[f"delta.{name}-{baseline}.{metric}"] = {"mean": float(np.mean(diffs)), "ci95": [low, high]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "papers": results}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))  # noqa: T201


if __name__ == "__main__":
    main()
