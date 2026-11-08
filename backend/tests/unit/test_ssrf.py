from __future__ import annotations

import httpx
import pytest

from mezvuat.core.errors import PayloadTooLargeError, UnsupportedMediaError, UrlRejectedError
from mezvuat.ingestion.ssrf import fetch_url, validate_url


def resolver_for(mapping: dict[str, list[str]]):
    def resolve(host: str, port: int) -> list[str]:
        return mapping[host]

    return resolve


PUBLIC = resolver_for({"example.com": ["93.184.216.34"]})


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "gopher://example.com",
        "http://user:pass@example.com/",
        "http://metadata.google.internal/computeMetadata/v1/",
    ],
)
def test_disallowed_schemes_and_forms(url: str) -> None:
    with pytest.raises(UrlRejectedError):
        validate_url(url, PUBLIC)


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.5",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",
        "0.0.0.0",  # noqa: S104
        "::1",
        "fe80::1",
        "fc00::1",
        "::ffff:127.0.0.1",
    ],
)
def test_non_public_addresses_are_rejected(address: str) -> None:
    resolver = resolver_for({"evil.test": [address]})
    with pytest.raises(UrlRejectedError) as caught:
        validate_url("http://evil.test/", resolver)
    assert caught.value.code == "url_blocked_address"


def test_one_private_address_among_public_ones_is_rejected() -> None:
    resolver = resolver_for({"mixed.test": ["93.184.216.34", "10.0.0.1"]})
    with pytest.raises(UrlRejectedError):
        validate_url("http://mixed.test/", resolver)


def test_public_url_is_accepted() -> None:
    target = validate_url("https://example.com/a?b=1", PUBLIC)
    assert (target.host, target.port, target.address, target.path_and_query) == (
        "example.com",
        443,
        "93.184.216.34",
        "/a?b=1",
    )


def make_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_connects_to_validated_ip_and_keeps_host() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, content=b"<html>x</html>")

    fetched = fetch_url(
        "https://example.com/page", client=make_client(handler), timeout_s=1, max_bytes=1000, resolver=PUBLIC
    )
    assert fetched.content_type == "text/html"
    assert seen[0].url.host == "93.184.216.34"
    assert seen[0].headers["host"] == "example.com"
    assert seen[0].extensions["sni_hostname"] == "example.com"


def test_redirect_to_private_address_is_rejected() -> None:
    resolver = resolver_for({"example.com": ["93.184.216.34"], "internal.test": ["10.1.2.3"]})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://internal.test/admin"})

    with pytest.raises(UrlRejectedError) as caught:
        fetch_url("https://example.com/", client=make_client(handler), timeout_s=1, max_bytes=100, resolver=resolver)
    assert caught.value.code == "url_blocked_address"


def test_too_many_redirects() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://example.com/again"})

    with pytest.raises(UrlRejectedError) as caught:
        fetch_url("https://example.com/", client=make_client(handler), timeout_s=1, max_bytes=100, resolver=PUBLIC)
    assert caught.value.code == "url_too_many_redirects"


def test_size_cap_and_content_type() -> None:
    def large(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"x" * 2000)

    with pytest.raises(PayloadTooLargeError):
        fetch_url("https://example.com/", client=make_client(large), timeout_s=1, max_bytes=100, resolver=PUBLIC)

    def binary(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/zip"}, content=b"PK")

    with pytest.raises(UnsupportedMediaError):
        fetch_url("https://example.com/", client=make_client(binary), timeout_s=1, max_bytes=100, resolver=PUBLIC)
