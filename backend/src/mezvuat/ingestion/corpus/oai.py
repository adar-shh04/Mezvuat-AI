"""arXiv OAI-PMH harvesting (metadata only). arXiv's terms: one request every three seconds, one connection."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from xml.etree.ElementTree import Element

import defusedxml.ElementTree as SafeET
import httpx

OAI_BASE_URL = "https://oaipmh.arxiv.org/oai"
API_BASE_URL = "https://export.arxiv.org/api/query"
USER_AGENT = "mezvuat-ai-corpus-builder/0.1 (research prototype)"
MIN_INTERVAL_S = 3.0

_OAI = "{http://www.openarchives.org/OAI/2.0/}"
_RAW = "{http://arxiv.org/OAI/arXivRaw/}"
_ATOM = "{http://www.w3.org/2005/Atom}"
_VERSION_SUFFIX = re.compile(r"v\d+$")


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


def _text(element: Element, tag: str) -> str:
    found = element.find(f"{_RAW}{tag}")
    return " ".join((found.text or "").split()) if found is not None else ""


def _parse_date(value: str) -> date:
    parsed = parsedate_to_datetime(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return datetime.fromtimestamp(parsed.timestamp(), UTC).date()


def parse_record(raw: Element) -> ArxivRecord:
    versions: list[ArxivVersion] = []
    for version in raw.findall(f"{_RAW}version"):
        label = version.attrib["version"]
        date_text = _text(version, "date")
        versions.append(ArxivVersion(label=label, ordinal=int(label.lstrip("v")), submitted=_parse_date(date_text)))
    versions.sort(key=lambda v: v.ordinal)
    return ArxivRecord(
        arxiv_id=_text(raw, "id"),
        title=_text(raw, "title"),
        authors=_text(raw, "authors"),
        categories=tuple(_text(raw, "categories").split()),
        license=_text(raw, "license") or None,
        abstract=_text(raw, "abstract"),
        versions=tuple(versions),
    )


def parse_get_record(xml_bytes: bytes) -> ArxivRecord | None:
    """Parse an OAI GetRecord response. Returns None for a record that does not exist or was deleted."""
    root = SafeET.fromstring(xml_bytes)
    error = root.find(f"{_OAI}error")
    if error is not None:
        if error.attrib.get("code") == "idDoesNotExist":
            return None
        raise RuntimeError(f"OAI-PMH error {error.attrib.get('code')}: {error.text}")
    record = root.find(f"{_OAI}GetRecord/{_OAI}record")
    if record is None:
        raise RuntimeError("OAI-PMH response has no GetRecord element")
    header = record.find(f"{_OAI}header")
    if header is not None and header.attrib.get("status") == "deleted":
        return None
    raw = record.find(f"{_OAI}metadata/{_RAW}arXivRaw")
    return parse_record(raw) if raw is not None else None


def parse_search_ids(xml_bytes: bytes) -> list[str]:
    """Extract versionless arXiv ids from an arXiv API Atom feed."""
    root = SafeET.fromstring(xml_bytes)
    ids: list[str] = []
    for entry in root.findall(f"{_ATOM}entry"):
        element = entry.find(f"{_ATOM}id")
        if element is not None and element.text:
            ids.append(_VERSION_SUFFIX.sub("", element.text.rsplit("/abs/", 1)[-1].strip()))
    return ids


def _get(client: httpx.Client, limiter: RateLimiter, url: str, params: dict[str, str]) -> httpx.Response:
    """One polite request, retrying transient arXiv failures (it returns 503 and timeouts under load)."""
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


def search_ids(
    query: str, client: httpx.Client, limiter: RateLimiter, *, max_results: int, page_size: int = 200
) -> Iterator[str]:
    """Yield candidate ids from the arXiv search API, newest first."""
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
        ids = parse_search_ids(response.content)
        if not ids:
            return
        yield from ids


def fetch_record(arxiv_id: str, client: httpx.Client, limiter: RateLimiter) -> ArxivRecord | None:
    """OAI ListRecords times out at arXiv (HTTP 503 after about 120 s), so records are fetched one by one."""
    response = _get(
        client,
        limiter,
        OAI_BASE_URL,
        {"verb": "GetRecord", "identifier": f"oai:arXiv.org:{arxiv_id}", "metadataPrefix": "arXivRaw"},
    )
    return parse_get_record(response.content)
