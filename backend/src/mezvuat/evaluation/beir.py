"""Load BEIR-format datasets. Files are untrusted data: only ever parsed as JSON lines and TSV."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx

BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"


@dataclass(frozen=True)
class BeirDoc:
    doc_id: str
    title: str
    text: str

    @property
    def full_text(self) -> str:
        return f"{self.title}\n{self.text}" if self.title else self.text


@dataclass(frozen=True)
class BeirDataset:
    name: str
    corpus: dict[str, BeirDoc]
    queries: dict[str, str]
    qrels: dict[str, dict[str, dict[str, int]]]  # split -> query id -> doc id -> grade


def download_dataset(name: str, root: Path) -> tuple[Path, str]:
    """Download and unpack into its own directory. Returns the dataset dir and the zip's SHA-256."""
    target = root / name
    digest_file = root / f"{name}.zip.sha256"
    if (target / "corpus.jsonl").exists() and digest_file.exists():
        return target, digest_file.read_text(encoding="ascii").strip()
    root.mkdir(parents=True, exist_ok=True)
    response = httpx.get(f"{BEIR_BASE_URL}/{name}.zip", timeout=600.0, follow_redirects=True)
    response.raise_for_status()
    digest = hashlib.sha256(response.content).hexdigest()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        for member in archive.namelist():
            # Extract only the expected data files, and never honour paths from the archive.
            parts = Path(member).parts
            if member.endswith((".jsonl", ".tsv")) and len(parts) >= 2 and ".." not in parts:
                destination = target.joinpath(*parts[1:])
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(member))
    digest_file.write_text(digest, encoding="ascii")
    return target, digest


def load_dataset(directory: Path, name: str) -> BeirDataset:
    corpus: dict[str, BeirDoc] = {}
    with (directory / "corpus.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            corpus[str(row["_id"])] = BeirDoc(str(row["_id"]), row.get("title") or "", row.get("text") or "")
    queries: dict[str, str] = {}
    with (directory / "queries.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            queries[str(row["_id"])] = row["text"]
    qrels: dict[str, dict[str, dict[str, int]]] = {}
    for path in sorted((directory / "qrels").glob("*.tsv")):
        split: dict[str, dict[str, int]] = {}
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            next(reader)
            for query_id, doc_id, score in reader:
                if str(doc_id) in corpus and str(query_id) in queries:
                    split.setdefault(str(query_id), {})[str(doc_id)] = int(score)
        qrels[path.stem] = split
    return BeirDataset(name=name, corpus=corpus, queries=queries, qrels=qrels)


def split_dev_from_train(
    train_qrels: dict[str, dict[str, int]], *, dev_fraction: float = 0.2, seed: int = 13
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    """Carve a tuning split out of the training queries. Test queries are never touched by tuning."""
    ids = sorted(train_qrels)
    random.Random(seed).shuffle(ids)  # noqa: S311  (reproducible split, not security)
    cut = int(len(ids) * dev_fraction)
    dev = {qid: train_qrels[qid] for qid in ids[:cut]}
    remaining = {qid: train_qrels[qid] for qid in ids[cut:]}
    return remaining, dev
