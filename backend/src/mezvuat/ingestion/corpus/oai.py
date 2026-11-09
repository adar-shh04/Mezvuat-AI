"""arXiv corpus metadata.

The search API supplies titles, authors, categories and abstracts; the abstract page supplies licence and version
history, because arXiv OAI-PMH timed out on both ListRecords and GetRecord when this was built (HTTP 503 after about
120 s). arXiv terms: at most one request every three seconds, on one connection.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import defusedxml.ElementTree as SafeET
import httpx

API_BASE_URL = "https://export.arxiv.org/api/query"
ABS_BASE_URL = "https://arxiv.org/abs"
USER_AGENT = "mezvuat-ai-corpus-builder/0.1 (research prototype)"
MIN_INTERVAL_S = 3.0

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"
_VERSION_SUFFIX = re.compile(r"v\d+$")
_LICENSE_LINK = re.compile(r'class="abs-license">\s*<a href="([^"]+)"')
_VERSION_ROW = re.compile(
    r"\[v(\d+)\](?:\s*</(?:a|strong)>)+\s*([A-Z][a-z]{2}, \d{1,2} [A-Z][a-z]{2} \d{4} [\d:]{8} [A-Z]{3})"
)
    r"\[v(\d+)\]\s*</(?:a|strong)>\s*([A-Z][a-z]{2}, \d{1,2} [A-Z][a-z]{2} \d{4} [\d:]{8} [A-Z]{3})"
)


class RateLimiter:
    """Blocks so that consecutive `wait()` calls are at least `min_interval_s` apart."""

    def __init__(
        self,
        min_interval_s: float = MIN_INTERVAL_S,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._min_interval_s = min_interval_s
        self._monotonic = monotonic
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None:
            remaining = self._min_interval_s - (self._monotonic() - self._last)
            if remaining > 0:
                self._sleep(remaining)
        self._last = self._monotonic()


@dataclass(frozen=True)
class ArxivVersion:
    label: str
    ordinal: int
    submitted: date


@dataclass(frozen=True)
class ArxivRecord:
    arxiv_id: str
    title: str
    authors: str
    categories: tuple[str, ...]
    license: str | None
    abstract: str
    versions: tuple[ArxivVersion, ...]

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["categories"] = list(self.categories)
        data["versions"] = [
            {"label": v.label, "ordinal": v.ordinal, "submitted": v.submitted.isoformat()} for v in self.versions
        ]
        return data


def _parse_date(value: str) -> date:
    parsed = parsedate_to_datetime(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return datetime.fromtimestamp(parsed.timestamp(), UTC).date()


def _clean(element: Any) -> str:
    return " ".join((element.text or "").split()) if element is not None else ""


@dataclass(frozen=True)
class SearchEntry:
    arxiv_id: str
    title: str
    authors: str
    categories: tuple[str, ...]
    abstract: str


def parse_search_entries(xml_bytes: bytes) -> list[SearchEntry]:
    """Extract paper metadata from an arXiv API Atom feed."""
    root = SafeET.fromstring(xml_bytes)
    entries: list[SearchEntry] = []
    for entry in root.findall(f"{_ATOM}entry"):
        identifier = _clean(entry.find(f"{_ATOM}id"))
        if "/abs/" not in identifier:
            continue
        primary = entry.find(f"{_ARXIV}primary_category")
        others = [c.attrib.get("term", "") for c in entry.findall(f"{_ATOM}category")]
        first = primary.attrib.get("term", "") if primary is not None else ""
        categories = tuple(dict.fromkeys([first, *others]) if first else dict.fromkeys(others))
        entries.append(
            SearchEntry(
                arxiv_id=_VERSION_SUFFIX.sub("", identifier.rsplit("/abs/", 1)[-1]),
                title=_clean(entry.find(f"{_ATOM}title")),
                authors=", ".join(_clean(a.find(f"{_ATOM}name")) for a in entry.findall(f"{_ATOM}author")),
                categories=tuple(c for c in categories if c),
                abstract=_clean(entry.find(f"{_ATOM}summary")),
            )
        )
    return entries


def parse_abs_page(html: str) -> tuple[str | None, tuple[ArxivVersion, ...]]:
    """Licence URL (None means the default non-exclusive licence) and version history from an abstract page."""
    match = _LICENSE_LINK.search(html)
    license_url = match.group(1) if match else None
    versions = sorted(
        (
            ArxivVersion(label=f"v{number}", ordinal=int(number), submitted=_parse_date(stamp))
            for number, stamp in _VERSION_ROW.findall(html)
        ),
        key=lambda v: v.ordinal,
    )
    return license_url, tuple(versions)


def _get(client: httpx.Client, limiter: RateLimiter, url: str, params: dict[str, str] | None = None) -> httpx.Response:
    """One polite request, retrying transient arXiv failures."""
    for attempt in range(4):
        limiter.wait()
        try:
            response = client.get(url, params=params, timeout=60.0)
        except httpx.TransportError:
            if attempt == 3:
                raise
            continue
        if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
            time.sleep(min(int(response.headers.get("Retry-After", "20")), 120))
            continue
        response.raise_for_status()
        return response
    raise RuntimeError("unreachable")


def search_entries(
    query: str, client: httpx.Client, limiter: RateLimiter, *, max_results: int, page_size: int = 200
) -> Iterator[SearchEntry]:
    """Yield candidates from the arXiv search API, newest first."""
    for start in range(0, max_results, page_size):
        response = _get(
            client,
            limiter,
            API_BASE_URL,
            {
                "search_query": query,
                "start": str(start),
                "max_results": str(page_size),
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
        )
        entries = parse_search_entries(response.content)
        if not entries:
            return
        yield from entries


def fetch_record(entry: SearchEntry, client: httpx.Client, limiter: RateLimiter) -> ArxivRecord | None:
    """Complete a search entry with its licence and versions. None when no version history can be read."""
    response = _get(client, limiter, f"{ABS_BASE_URL}/{entry.arxiv_id}")
    license_url, versions = parse_abs_page(response.text)
    if not versions:
        return None
    return ArxivRecord(
        arxiv_id=entry.arxiv_id,
        title=entry.title,
        authors=entry.authors,
        categories=entry.categories,
        license=license_url,
        abstract=entry.abstract,
        versions=versions,
    )
