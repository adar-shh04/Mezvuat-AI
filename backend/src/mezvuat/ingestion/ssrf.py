"""SSRF-safe URL fetching (OWASP guidance): one DNS resolution, connect to the validated IP, re-check every hop."""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from mezvuat.core.errors import PayloadTooLargeError, UnsupportedMediaError, UrlRejectedError

ALLOWED_CONTENT_TYPES = frozenset({"text/html", "application/pdf", "text/plain", "application/xhtml+xml"})
MAX_REDIRECTS = 3
_METADATA_HOSTS = frozenset({"metadata.google.internal", "metadata", "instance-data"})

Resolver = Callable[[str, int], list[str]]


def system_resolver(host: str, port: int) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise UrlRejectedError("host name does not resolve", code="url_unresolvable") from exc
    return sorted({str(info[4][0]) for info in infos})


def _is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


@dataclass(frozen=True)
class ValidatedTarget:
    scheme: str
    host: str
    port: int
    address: str
    path_and_query: str


def validate_url(url: str, resolver: Resolver = system_resolver) -> ValidatedTarget:
    """Reject anything but http(s) to a public address. All resolved addresses must be public."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise UrlRejectedError("only http and https URLs are allowed")
    if parts.username or parts.password:
        raise UrlRejectedError("URLs with credentials are not allowed")
    host = parts.hostname
    if not host or host.lower() in _METADATA_HOSTS:
        raise UrlRejectedError("host is not allowed")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    addresses = resolver(host, port)
    if not addresses:
        raise UrlRejectedError("host name does not resolve", code="url_unresolvable")
    for address in addresses:
        if not _is_public(address):
            raise UrlRejectedError("host resolves to a non-public address", code="url_blocked_address")
    path = urlunsplit(("", "", parts.path or "/", parts.query, ""))
    return ValidatedTarget(scheme=parts.scheme, host=host, port=port, address=addresses[0], path_and_query=path)


@dataclass(frozen=True)
class FetchedUrl:
    final_url: str
    content_type: str
    body: bytes


def fetch_url(
    url: str,
    *,
    client: httpx.Client,
    timeout_s: float,
    max_bytes: int,
    resolver: Resolver = system_resolver,
) -> FetchedUrl:
    """Fetch with manual redirects, connecting to the validated IP while keeping Host and SNI."""
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        target = validate_url(current, resolver)
        netloc = f"[{target.address}]" if ":" in target.address else target.address
        pinned = f"{target.scheme}://{netloc}:{target.port}{target.path_and_query}"
        request = client.build_request(
            "GET",
            pinned,
            headers={"Host": target.host, "User-Agent": "mezvuat-ai-ingest/0.1"},
            timeout=timeout_s,
            extensions={"sni_hostname": target.host},
        )
        response = client.send(request, stream=True, follow_redirects=False)
        try:
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise UrlRejectedError("redirect without a location")
                current = urljoin(current, location)
                continue
            if response.status_code != 200:
                raise UrlRejectedError(f"server returned {response.status_code}", code="url_fetch_failed")
            content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
            if content_type not in ALLOWED_CONTENT_TYPES:
                raise UnsupportedMediaError(f"content type {content_type!r} is not supported")
            body = bytearray()
            for piece in response.iter_bytes():
                body.extend(piece)
                if len(body) > max_bytes:
                    raise PayloadTooLargeError("remote document exceeds the size limit")
            return FetchedUrl(final_url=current, content_type=content_type, body=bytes(body))
        finally:
            response.close()
    raise UrlRejectedError("too many redirects", code="url_too_many_redirects")
