"""arXiv OAI-PMH harvesting (metadata only). arXiv's terms: one request every three seconds, one connection."""

from __future__ import annotations

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
USER_AGENT = "mezvuat-ai-corpus-builder/0.1 (research prototype)"
MIN_INTERVAL_S = 3.0

_OAI = "{http://www.openarchives.org/OAI/2.0/}"
_RAW = "{http://arxiv.org/OAI/arXivRaw/}"


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


def parse_list_records(xml_bytes: bytes) -> tuple[list[ArxivRecord], str | None]:
    """Return the records on one OAI page and the resumption token, if any."""
    root = SafeET.fromstring(xml_bytes)
    error = root.find(f"{_OAI}error")
    if error is not None:
        if error.attrib.get("code") == "noRecordsMatch":
            return [], None
        raise RuntimeError(f"OAI-PMH error {error.attrib.get('code')}: {error.text}")
    list_records = root.find(f"{_OAI}ListRecords")
    if list_records is None:
        raise RuntimeError("OAI-PMH response has no ListRecords element")
    records: list[ArxivRecord] = []
    for record in list_records.findall(f"{_OAI}record"):
        header = record.find(f"{_OAI}header")
        if header is not None and header.attrib.get("status") == "deleted":
            continue
        metadata = record.find(f"{_OAI}metadata")
        raw = metadata.find(f"{_RAW}arXivRaw") if metadata is not None else None
        if raw is not None:
            records.append(parse_record(raw))
    token_element = list_records.find(f"{_OAI}resumptionToken")
    token = (token_element.text or "").strip() if token_element is not None else ""
    return records, token or None


def harvest(
    set_spec: str, from_date: date, client: httpx.Client, limiter: RateLimiter | None = None
) -> Iterator[ArxivRecord]:
    """Yield every record in an OAI set updated since `from_date`, one rate-limited page at a time."""
    limiter = limiter or RateLimiter()
    params: dict[str, str] = {
        "verb": "ListRecords",
        "metadataPrefix": "arXivRaw",
        "set": set_spec,
        "from": from_date.isoformat(),
    }
    while True:
        limiter.wait()
        response = client.get(OAI_BASE_URL, params=params, timeout=120.0)
        if response.status_code == 503:
            retry_after = int(response.headers.get("Retry-After", "30"))
            time.sleep(min(retry_after, 300))
            continue
        response.raise_for_status()
        records, token = parse_list_records(response.content)
        yield from records
        if token is None:
            return
        params = {"verb": "ListRecords", "resumptionToken": token}
