"""BM25 lexical retrieval with bm25s (ADR-0004): one index per workspace and index version.

Isolation is physical: a query can only load its own workspace's index directory.
"""

from __future__ import annotations

import json
import shutil
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import bm25s
import numpy as np
import Stemmer

from mezvuat.retrieval.types import ChunkPayload, Filters, ScoredChunk

BM25_METHOD = "lucene"
BM25_K1 = 1.2
BM25_B = 0.75
_PAYLOADS_FILE = "chunks.json"


@dataclass(frozen=True)
class Bm25Params:
    method: str = BM25_METHOD
    k1: float = BM25_K1
    b: float = BM25_B


def tokenize(texts: Sequence[str]) -> list[list[str]]:
    """Lowercase, remove English stopwords, Snowball-stem. The same analysis is used for documents and queries."""
    stemmer = Stemmer.Stemmer("english")
    tokens = bm25s.tokenize(
        list(texts), lower=True, stopwords="en", stemmer=stemmer, return_ids=False, show_progress=False
    )
    return [list(row) for row in tokens]


def payload_allowed(payload: ChunkPayload, filters: Filters) -> bool:
    if filters.document_ids and payload.document_id not in filters.document_ids:
        return False
    if filters.latest_only and payload.is_latest_version is not True:
        return False
    if payload.section_kind in filters.exclude_section_kinds:
        return False
    if filters.published_from or filters.published_to:
        if payload.published_at is None:
            # Unknown dates never satisfy a date filter (ADR-0010).
            return False
        published = payload.published_at[:10]
        if filters.published_from and published < filters.published_from.isoformat():
            return False
        if filters.published_to and published > filters.published_to.isoformat():
            return False
    return True


class Bm25Index:
    def __init__(self, retriever: bm25s.BM25, payloads: list[ChunkPayload]) -> None:
        self._retriever = retriever
        self._payloads = payloads

    @classmethod
    def build(cls, payloads: list[ChunkPayload], texts: Sequence[str], params: Bm25Params | None = None) -> Bm25Index:
        if len(payloads) != len(texts):
            raise ValueError("payloads and texts must align")
        if not texts:
            raise ValueError("cannot build a BM25 index over no chunks")
        params = params or Bm25Params()
        retriever = bm25s.BM25(method=params.method, k1=params.k1, b=params.b)
        retriever.index(tokenize(texts), show_progress=False)
        return cls(retriever, payloads)

    def __len__(self) -> int:
        return len(self._payloads)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._retriever.save(str(directory))
        (directory / _PAYLOADS_FILE).write_text(json.dumps([asdict(p) for p in self._payloads]), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> Bm25Index:
        retriever = bm25s.BM25.load(str(directory), mmap=False)
        raw = json.loads((directory / _PAYLOADS_FILE).read_text(encoding="utf-8"))
        return cls(retriever, [ChunkPayload(**item) for item in raw])

    def search(self, query: str, k: int, filters: Filters | None = None) -> list[ScoredChunk]:
        filters = filters or Filters()
        tokens = tokenize([query])[0]
        if not tokens or not self._payloads:
            return []
        mask = np.fromiter(
            (payload_allowed(p, filters) for p in self._payloads), dtype=np.float32, count=len(self._payloads)
        )
        if not mask.any():
            return []
        limit = min(k, len(self._payloads))
        results, scores = self._retriever.retrieve(
            [tokens], k=limit, weight_mask=mask, show_progress=False, sorted=True
        )
        hits = [
            (self._payloads[int(row)].chunk_id, float(score))
            for row, score in zip(results[0], scores[0], strict=True)
            if score > 0
        ]
        hits.sort(key=lambda item: (-item[1], item[0]))
        return [ScoredChunk(chunk_id=chunk_id, score=score) for chunk_id, score in hits]


class WorkspaceLexicalIndexStore:
    """Writes versioned indices atomically and serves them through a small LRU."""

    def __init__(
        self,
        root: Path,
        current_version: Callable[[str], int | None],
        max_loaded: int = 8,
    ) -> None:
        self._root = root
        self._current_version = current_version
        self._max_loaded = max_loaded
        self._loaded: OrderedDict[tuple[str, int], Bm25Index] = OrderedDict()
        self._lock = threading.Lock()

    def _dir(self, workspace_id: str, version: int) -> Path:
        return self._root / workspace_id / f"v{version}"

    def write(self, workspace_id: str, version: int, payloads: list[ChunkPayload], texts: Sequence[str]) -> None:
        """Build into a temporary directory and rename into place; the pointer is swapped by the caller."""
        index = Bm25Index.build(payloads, texts)
        temp = self._root / workspace_id / f"tmp-{uuid.uuid4().hex}"
        try:
            index.save(temp)
            temp.replace(self._dir(workspace_id, version))
        except BaseException:
            shutil.rmtree(temp, ignore_errors=True)
            raise

    def prune(self, workspace_id: str, keep_version: int) -> None:
        workspace_dir = self._root / workspace_id
        if not workspace_dir.exists():
            return
        for child in workspace_dir.iterdir():
            if child.name != f"v{keep_version}":
                shutil.rmtree(child, ignore_errors=True)

    def delete_workspace(self, workspace_id: str) -> None:
        shutil.rmtree(self._root / workspace_id, ignore_errors=True)
        with self._lock:
            for key in [key for key in self._loaded if key[0] == workspace_id]:
                del self._loaded[key]

    def _get(self, workspace_id: str, version: int) -> Bm25Index:
        key = (workspace_id, version)
        with self._lock:
            if key in self._loaded:
                self._loaded.move_to_end(key)
                return self._loaded[key]
        index = Bm25Index.load(self._dir(workspace_id, version))
        with self._lock:
            self._loaded[key] = index
            for stale in [other for other in self._loaded if other[0] == workspace_id and other != key]:
                del self._loaded[stale]
            while len(self._loaded) > self._max_loaded:
                self._loaded.popitem(last=False)
        return index

    def search(self, workspace_id: str, query: str, k: int, filters: Filters) -> list[ScoredChunk]:
        version = self._current_version(workspace_id)
        if version is None:
            return []
        return self._get(workspace_id, version).search(query, k, filters)
